import weakref
import sys
import torch

sys.path.append(".")

from tools.visualization_utils import *

from scripts.workflows.automatic_articulation.utils.process_action import curobo2robot_actions, process_action
from scripts.workflows.automatic_articulation.utils.grasp_sampler import GraspSampler


class GrasperEnv:

    def __init__(
        self,
        env,
        planner,
        use_relative_pose=False,
        collision_checker=True,
        env_config=None,
    ):

        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.kitchen = env.scene["kitchen"]
        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker
        self.env_config = env_config

        self.grasp_sampler = GraspSampler(
            env,
            "source/config/task/automatic_articulation/m2t2_config.yaml",
            "../M2T2/m2t2.pth",
            env_config=env_config)

        self.reverse_sample = False
        self.planner = planner

        self.count_steps = 0

        self.init_setting()

    def init_setting(self):
        self.lift_or_not = False
        self.init_grasp_object_state = torch.zeros(1, 3).to(self.device)

        self.target_handle_name = self.env.scene[
            "kitchen"].cfg.articulation_cfg["target_drawer"]
        self.robot_offset = torch.as_tensor(
            self.env.scene["kitchen"].cfg.articulation_cfg[
                "robot_random_range"][self.target_handle_name]["offset"]).to(
                    self.device)
        self.handle_id, handle_name = self.env.scene["kitchen"].find_bodies(
            self.target_handle_name)
        self.joint_ids, joint_names = self.kitchen.find_joints(
            self.kitchen.cfg.articulation_cfg["robot_random_range"][
                self.target_handle_name]["joint_name"])

        self.lift_threshold = self.env_config["params"]["Task"]["grasper"][
            "lift_threshold"]

        self.planner_tolerance_iter = self.env_config["params"]["Task"][
            "grasper"]["planner_tolerance_iter"]

        self.action_horiozon = self.env_config["params"]["Task"]["grasper"]
        self.grasp_object = self.env.scene[
            self.env.scene["kitchen"].cfg.articulation_cfg["target_object"]]
        self.grasp_object_name = self.env.scene[
            "kitchen"].cfg.articulation_cfg["target_object"]

        self.isaac2m2t2_quat = torch.as_tensor(
            self.env_config["params"]["Task"]["grasper"]["isaac2m2t2_offset"]
            [3:7]).to(self.device).unsqueeze(0)
        self.isaac2m2t2_trans = torch.as_tensor(
            self.env_config["params"]["Task"]["grasper"]["isaac2m2t2_offset"]
            [:3]).to(self.device).unsqueeze(0)

        self.gripper_offset = torch.tensor(
            self.env_config["params"]["Task"]["grasper"]["gripper_offset"]).to(
                self.device)
        self.m2t2_gripper_offset = torch.tensor(
            self.env_config["params"]["Task"]["grasper"]
            ["m2t2_gripper_offset"]).to(self.device)

        self.grasp_angle_range = self.env_config["params"]["Task"]["grasper"][
            "grasp_angle_range"]

        self.init_horizon()

    def init_horizon(self):

        self.approach_horizon = self.action_horiozon["approach_horizon"]
        self.close_gripper_horizon = self.action_horiozon["close_horizon"]
        self.lift_object_horizon = self.action_horiozon["lift_horizon"]
        self.lift_speed = self.action_horiozon["lift_speed"]

        self.pre_lift_offet = self.approach_horizon + self.close_gripper_horizon + self.lift_object_horizon
        curobo_ik = self.planner.curobo_ik
        self.init_ee_pose = torch.as_tensor(
            self.action_horiozon["init_ee_pos"]).to(self.device).clone()
        self.init_jpos = curobo_ik.plan_motion(
            self.init_ee_pose[:3].clone(),
            self.init_ee_pose[3:7].clone()).to(self.device)
        self.init_ee_pose = self.init_ee_pose.unsqueeze(0)

    def filter_by_confidence(self, grasps, grasp_confidence, top_k=1000):
        top_k = min(top_k, len(grasp_confidence))

        top_confidence_values, top_indices = torch.topk(
            grasp_confidence, top_k)
        return grasps[top_indices], top_confidence_values

    def filter_by_dist(self, top_grasp_location, top_grasp_num=20):

        dist = torch.linalg.norm(top_grasp_location[:, :2] -
                                 top_grasp_location[0][:2],
                                 dim=1)

        return torch.argsort(dist)[:top_grasp_num]

    def choose_best_pos(self, grasps, grasp_confidence, top_k=100):

        top_grasp, _ = self.filter_by_confidence(grasps, grasp_confidence,
                                                 top_k)
        top_grasp_location = top_grasp[:, :3, 3]
        top_grasp_quat = math_utils.quat_from_matrix(top_grasp[:, :3, :3])

        top_grasp_location, top_grasp_quat = math_utils.combine_frame_transforms(
            top_grasp_location, top_grasp_quat,
            self.isaac2m2t2_trans.repeat(len(top_grasp_location), 1),
            self.isaac2m2t2_quat.repeat(len(top_grasp_location), 1))

        top_indices = self.filter_by_dist(top_grasp_location)

        return top_grasp_location[top_indices], top_grasp_quat[
            top_indices], top_indices

    def sample_gripper_pose(self, observation):

        while True:

            self.m2t2_data, self.m2t2_outputs = self.grasp_sampler.load_and_predict(
                observation["policy"])

            if len(self.m2t2_outputs['grasps']) == 0:
                continue
            grasps = self.m2t2_outputs['grasps'][0]
            grasp_confidence = self.m2t2_outputs['grasp_confidence'][0]
            if grasps.shape[0] >= 100:
                print("number of sampler grasps", grasps.shape[0])
                break
        m2t2_ee_pos, m2t2_ee_quat, top_indices = self.choose_best_pos(
            grasps, grasp_confidence)
        self.grasp_sampler.visualization_select_pose(
            self.m2t2_data, self.m2t2_outputs,
            [self.m2t2_outputs['grasps'][0][top_indices.cpu()]])

        # clip the euler angle range
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(m2t2_ee_quat)

        roll[:] = torch.clamp(roll[:], self.grasp_angle_range[0],
                              self.grasp_angle_range[1])  # 180 degree
        m2t2_ee_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # get curobo planning pose
        curobo_pose_b = m2t2_ee_pos.clone()
        curobo_quat_b = m2t2_ee_quat.clone()
        curobo_quat_b = math_utils.shortest_angles(curobo_quat_b, self.env)
        m2t2_ee_quat = math_utils.shortest_angles(m2t2_ee_quat, self.env)

        curobo_pose_b, curobo_quat_b = math_utils.combine_frame_transforms(
            m2t2_ee_pos, m2t2_ee_quat, self.gripper_offset[:3].unsqueeze(0).to(
                self.device).repeat_interleave(len(m2t2_ee_pos), 0),
            self.gripper_offset[3:7].unsqueeze(0).repeat_interleave(
                len(m2t2_ee_pos), 0))

        # add the offset for the controller
        target_robot_positions, target_robot_quaternion = math_utils.combine_frame_transforms(
            m2t2_ee_pos, m2t2_ee_quat,
            self.m2t2_gripper_offset[:3].unsqueeze(0).to(
                self.device).repeat_interleave(len(m2t2_ee_pos), 0),
            self.m2t2_gripper_offset[3:7].unsqueeze(0).repeat_interleave(
                len(m2t2_ee_pos), 0))
        self.all_robot_pose = torch.cat(
            [target_robot_positions, target_robot_quaternion], dim=1)
        self.curobo_pos = torch.cat([curobo_pose_b, curobo_quat_b], dim=1)
        return True

    def plan_for_grasp(self, observation):

        robot_qpos = observation["policy"]["joint_pos"]

        planner_count = 0

        if self.collision_checker:
            self.planner.add_obstacle(
                plan_grasp=True,
                target_object_name=self.env.scene["kitchen"].cfg.
                articulation_cfg["target_object"])

        index = 0
        ee_pose = None
        if self.reverse_sample:
            indices = list(range(len(self.curobo_pos)))
            indices.reverse()
        else:
            indices = list(range(len(self.curobo_pos)))

        # Shuffle the indices
        shuffled_indices = torch.randperm(len(indices)).tolist()

        planner_count = 0

        for index in shuffled_indices:
            ee_pose, traj = self.planner.plan_motion(
                robot_qpos,
                self.curobo_pos[index, :3].unsqueeze(0),
                self.curobo_pos[index, 3:].unsqueeze(0),
            )

            self.robot_pose = self.all_robot_pose[index]
            planner_count += 1

            # Remove the sampled index from shuffled_indices
            shuffled_indices.remove(index)

            if ee_pose is not None or planner_count > self.planner_tolerance_iter:
                break

        return ee_pose, index

    def reset_planner_grasp(
        self,
        index,
        ee_pose,
        sample_grasp=True,
    ):
        self.planner.clear_obstacles()
        if sample_grasp:
            if self.reverse_sample:
                self.all_robot_pose = self.all_robot_pose[:index]
                self.curobo_pos = self.curobo_pos[:index]
            else:
                self.all_robot_pose = self.all_robot_pose[index + 1:]
                self.curobo_pos = self.curobo_pos[index + 1:]
            self.reverse_sample = not self.reverse_sample

        else:

            robot_qpos = self.robot.root_physx_view.get_dof_positions()
            ee_pose, traj = self.planner.plan_motion(
                robot_qpos,
                ee_pose[:3].unsqueeze(0),
                ee_pose[3:7].unsqueeze(0),
            )
            if ee_pose is None:
                return None
        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion

        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)

        _, self.target_ee_traj = curobo2robot_actions(curobo_target_ee_pos,
                                                      self.device)

        # approach the target pose
        approach_ee_pose = torch.cat([
            self.robot_pose.unsqueeze(0),
            torch.ones((self.env.num_envs, 1)).to(self.device) * 1
        ],
                                     dim=1).repeat(self.approach_horizon, 1)

        close_gripper_ee_pose = torch.cat([
            self.robot_pose.unsqueeze(0),
            torch.ones((self.env.num_envs, 1)).to(self.device) * -1
        ],
                                          dim=1).repeat(
                                              self.close_gripper_horizon, 1)
        lift_ee_pose = self.lift_object()

        self.target_ee_traj = torch.cat([
            self.target_ee_traj, approach_ee_pose, close_gripper_ee_pose,
            lift_ee_pose
        ], 0)

        self.reach_length = len(self.target_ee_traj)
        self.count_steps = 0

        self.lift_or_not = False
        return self.target_ee_traj

    def reset(self, observation):

        success = self.sample_gripper_pose(observation)
        if not success:
            return None

        ee_pose, index = self.plan_for_grasp(observation)

        if ee_pose is None:
            return None
        self.reset_planner_grasp(index, ee_pose)

        self.init_grasp_object_state = self.grasp_object._data.root_state_w[:, :
                                                                            3]

        success = (math_utils.euler_xyz_from_quat(
            observation["policy"][f"{self.grasp_object_name}_root_pose"]
            [:, 3:7])[1] > 0.15) & success
        return success

    def lift_object(self):

        lift_pose = self.robot_pose.clone().unsqueeze(0).repeat(
            self.lift_object_horizon, 1)

        lift_pose[:, 2] += torch.arange(self.lift_object_horizon).to(
            self.device) * self.lift_speed[2]

        lift_pose[:, 0] -= torch.arange(self.lift_object_horizon).to(
            self.device) * self.lift_speed[0]

        target_ee_pose = torch.cat([
            lift_pose,
            torch.ones((lift_pose.shape[0], 1)).to(self.device) * -1
        ],
                                   dim=1)
        return target_ee_pose

    def success_or_not(self, observation):
        if not self.lift_or_not:
            self.lift_or_not = self.grasp_object._data.root_state_w[
                0,
                2] > self.init_grasp_object_state[0, 2] + self.lift_threshold
        placement_or_not = abs(observation["policy"]["drawer_pose"][0, 2] -
                               observation["policy"]
                               [f"{self.grasp_object_name}_pose"][0, 2]) < 0.07

        return self.lift_or_not & placement_or_not  # need to check the placement

    def connect_policy(self, obs):

        if self.collision_checker:
            self.planner.add_obstacle(
                plan_grasp=True,
                target_object_name=self.env.scene["kitchen"].cfg.
                articulation_cfg["target_object"])
        target_ee_pose = self.init_ee_pose.clone()
        robot_qpos = self.robot.root_physx_view.get_dof_positions()

        ee_pose, _ = self.planner.plan_motion(
            robot_qpos,
            target_ee_pose[:, :3],
            target_ee_pose[:, 3:7],
        )

        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion

        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)

        _, self.target_connect_traj = curobo2robot_actions(
            curobo_target_ee_pos, self.device)
        # self.target_connect_traj = torch.cat(
        #     [leave_cabinet_pose, self.target_connect_traj], 0)

    # def step(self):

    #     if self.count_steps < self.reach_length:
    #         target_ee_pose = self.target_ee_traj[self.count_steps].unsqueeze(0)

    #     elif self.count_steps < self.reach_length + self.approach_horizon:

    #         target_ee_pose = torch.cat([
    #             self.robot_pose.unsqueeze(0),
    #             torch.ones((self.env.num_envs, 1)).to(self.device) * 1
    #         ],
    #                                    dim=1)

    #     elif self.count_steps < self.reach_length + self.approach_horizon + self.close_gripper_horizon:  #close gripper

    #         target_ee_pose = torch.cat([
    #             self.robot_pose.unsqueeze(0),
    #             torch.ones((self.env.num_envs, 1)).to(self.device) * -1
    #         ],
    #                                    dim=1)
    #     elif self.count_steps < self.reach_length + self.pre_lift_offet:  #lift up

    #         target_ee_pose = self.lift_object()

    #     else:

    #         lift_or_not = self.grasp_object._data.root_state_w[
    #             0,
    #             2] > self.init_grasp_object_state[0, 2] + self.lift_threshold

    #         return True, None, None, None, None, None, None, lift_or_not

    #     actions = process_action(target_ee_pose, self.use_relative_pose,
    #                              self.robot, self.device)
    #     actions[:, -1] = torch.sign(actions[:, -1] + 0.3)

    #     # actions = self.init_ee_pose
    #     observation, reward, terminate, time_out, info = self.env.step(actions)

    #     self.count_steps += 1
    #     return False, observation, reward, terminate, time_out, info, actions, False
