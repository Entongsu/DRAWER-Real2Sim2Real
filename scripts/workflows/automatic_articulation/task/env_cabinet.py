import weakref
import sys
import torch

sys.path.append(".")

from tools.visualization_utils import *

from scripts.workflows.automatic_articulation.utils.process_action import process_action, curobo2robot_actions


class CabinetOpenEnv:

    def __init__(self,
                 env,
                 planner,
                 use_relative_pose=False,
                 collision_checker=False,
                 env_config=None):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.kitchen = env.scene["kitchen"]
        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker

        self.env_config = env_config
        self.planner = planner

        self.init_setting()

        self.count_steps = 0

    def init_setting(self):

        # get drawer setting
        self.target_joint_type = self.kitchen.cfg.articulation_cfg[
            "target_joint_type"]

        self.target_handle_name = self.env.scene[
            "kitchen"].cfg.articulation_cfg["target_drawer"]
        self.robot_offset = torch.as_tensor(
            self.env.scene["kitchen"].cfg.articulation_cfg[
                "robot_random_range"][self.target_handle_name]["offset"]).to(
                    self.device)
        self.handle_id, self.handle_name = self.env.scene[
            "kitchen"].find_bodies(self.target_handle_name)

        self.joint_ids, self.joint_names = self.kitchen.find_joints(
            self.kitchen.cfg.articulation_cfg["robot_random_range"][
                self.target_handle_name]["joint_name"])

        self.action_horiozon = self.env_config["params"]["Task"][
            "cabinet_open"]
        self.gripper_length_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["gripper_length_offset"]).to(
                self.device)
        self.planner_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["cabinet_open"]
            ["planner_offset"]).to(self.device)
        self.open_offset = torch.as_tensor(self.env_config["params"]["Task"]
                                           ["cabinet_open"]["open_offset"]).to(
                                               self.device)
        cabinet_robot_pose = torch.as_tensor(
            [0.35, -0.0, 0.40, 0.0, 1.0, 0.0, 0.0]).to(self.device)

        init_robot_pose = self.env.scene["kitchen"].cfg.articulation_cfg[
            "robot_random_range"][self.target_handle_name]["init_robot_pose"]

        cabinet_robot_pose[:3] = torch.as_tensor(init_robot_pose["pos"]).to(
            self.device)
        delta_quat01 = obtain_target_quat_from_multi_angles(
            init_robot_pose["rot"]["axis"],
            init_robot_pose["rot"]["angles"]).to(self.device)
        cabinet_robot_pose[3:] = math_utils.quat_mul(delta_quat01,
                                                     cabinet_robot_pose[3:])

        #TODO: need to clean up
        # init robot pose
        curobo_ik = self.planner.curobo_ik
        # if self.target_joint_type == "revolute":

        #     cabinet_robot_pose = torch.as_tensor(
        #         [0.35, -0.0, 0.40, 0.0, 1.0, 0.0, 0.0]).to(self.device)

        #     delta_quat01 = obtain_target_quat_from_multi_angles([
        #         2,
        #         1,
        #         0,
        #     ], [np.pi / 2, -np.pi / 2, -np.pi / 2]).to(self.device)
        #     cabinet_robot_pose[3:] = math_utils.quat_mul(
        #         delta_quat01, cabinet_robot_pose[3:])

        # elif self.target_joint_type == "prismatic":

        #     cabinet_robot_pose = torch.as_tensor(
        #         [0.35, 0.0, 0.45, 0.0, 1.0, 0.0, 0.0]).to(self.device)

        #     delta_quat01 = obtain_target_quat_from_multi_angles(
        #         [2, 1, 0], [np.pi / 2, -np.pi / 2, np.pi]).to(self.device)
        #     cabinet_robot_pose[3:] = math_utils.quat_mul(
        #         delta_quat01, cabinet_robot_pose[3:])

        self.init_jpos = curobo_ik.plan_motion(
            cabinet_robot_pose[:3].clone(),
            cabinet_robot_pose[3:7].clone()).to(self.device)
        cabinet_robot_pose = cabinet_robot_pose.unsqueeze(0)

        self.init_ee_pose = torch.cat([
            cabinet_robot_pose[:, :3], cabinet_robot_pose[:, 3:7],
            torch.ones(1, 1).to(self.device)
        ],
                                      dim=1)

        self.cabinet_trajectory = []
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

    def sample_gripper_pose(self, ):

        # simulate environment
        robot_pose = self.init_ee_pose.clone()[0]

        # get drawer handle location
        handle_location = self.kitchen._data.body_state_w[0][
            self.handle_id][:, :3]
        handle_orientation = self.kitchen._data.body_state_w[0][
            self.handle_id,
        ][:, 3:7]

        # the ope direction of the drawer
        self.open_dir = math_utils.axis_angle_from_quat(
            math_utils.quat_mul(
                self.kitchen._data.root_state_w[0, 3:7],
                math_utils.quat_inv(
                    self.kitchen._data.default_root_state[0, 3:7])))[2]

        # calculate relative transformation between robot and handle
        root_pose = self.robot._data.root_state_w
        translate_robot_to_handle, quat_robot_to_handle = math_utils.subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], handle_location,
            handle_orientation)

        x_rotation, y_rotation, z_rotation = self.robot_offset[3:6]
        delta_quat02 = math_utils.quat_from_euler_xyz(
            x_rotation, y_rotation, self.open_dir).to(self.device)
        robot_pose[3:7] = math_utils.quat_mul(delta_quat02, robot_pose[3:7])
        robot_pose[:3] = translate_robot_to_handle[0, :3]
        robot_pose[:3] += self.robot_offset[:3]

        # get curobo planning pose
        curobo_pose_b, curobo_quat_b = math_utils.combine_frame_transforms(
            robot_pose[:3], robot_pose[3:7], self.planner_offset[:3],
            self.planner_offset[3:7])

        robot_pose[:3], robot_pose[3:7] = math_utils.combine_frame_transforms(
            robot_pose[:3], robot_pose[3:7], self.gripper_length_offset[:3],
            self.gripper_length_offset[3:7])

        return robot_pose[:7], handle_location, torch.cat(
            [curobo_pose_b, curobo_quat_b], dim=0)

    def reset(self, observation):

        ee_pose = None

        robot_qpos = observation["policy"]["joint_pos"]

        self.robot_pose, handle_location, curobo_pos = self.sample_gripper_pose(
        )

        if self.collision_checker:
            self.planner.add_obstacle()

        ee_pose, traj = self.planner.plan_motion(
            robot_qpos,
            curobo_pos[:3].unsqueeze(0),
            curobo_pos[3:].unsqueeze(0),
        )
        self.planner.clear_obstacles()

        if ee_pose is None:
            return None
        if ee_pose is not None:
            if len(ee_pose.ee_position
                   ) > 80:  # long trajectory, replan the motion
                return None

        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion
        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)[5:]
        _, self.target_ee_traj = curobo2robot_actions(curobo_target_ee_pos,
                                                      self.device)

        # refine the trajectory by adding the approach motion
        self.refine_approach()
        self.drawer_open()

        self.reach_length = len(self.target_ee_traj)

        self.count_steps = 0
        return observation

    def refine_approach(self):

        approach_steps = self.action_horiozon["approach_horizon"]
        approach_ee_pose = torch.cat([
            self.robot_pose.unsqueeze(0),
            torch.ones((self.env.num_envs, 1)).to(self.device)
        ],
                                     dim=1).repeat(approach_steps, 1)
        close_ee_pose = torch.cat([
            self.robot_pose.unsqueeze(0),
            torch.ones((self.env.num_envs, 1)).to(self.device) * -1
        ],
                                  dim=1).repeat(
                                      self.action_horiozon["close_horizon"], 1)
        self.target_ee_traj = torch.cat(
            [self.target_ee_traj, approach_ee_pose, close_ee_pose], dim=0)

    def drawer_open(self):
        target_ee_pose = torch.cat([
            self.robot_pose.unsqueeze(0).clone(),
            torch.ones((self.env.num_envs, 1)).to(self.device) * -1
        ],
                                   dim=1).repeat(
                                       self.action_horiozon["open_horizon"], 1)

        if self.target_joint_type == "prismatic":

            delta_x = self.action_horiozon["open_speed"] * torch.cos(
                self.open_dir)
            delta_y = self.action_horiozon["open_speed"] * torch.sin(
                self.open_dir)
            target_ee_pose[:, 0] -= torch.arange(
                self.action_horiozon["open_horizon"]).to(self.device) * delta_x
            target_ee_pose[:, 1] -= torch.arange(
                self.action_horiozon["open_horizon"]).to(self.device) * delta_y
        elif self.target_joint_type == "revolute":

            radius = self.robot_offset[-1]
            delta_time = torch.arange(self.action_horiozon["open_horizon"]).to(
                self.device)

            x_coords = radius * torch.cos(self.action_horiozon["open_speed"] *
                                          delta_time / 180 * torch.pi)
            y_coords = radius * torch.sin(self.action_horiozon["open_speed"] *
                                          delta_time / 180 * torch.pi)
            target_ee_pose[:, 0] -= (y_coords)  #* self.robot_offset[6]
            target_ee_pose[:, 1] += (radius - x_coords) * self.robot_offset[6]

            delta_quat = math_utils.quat_from_euler_xyz(
                torch.as_tensor(0.0), torch.as_tensor(0),
                torch.as_tensor(-self.robot_offset[6] *
                                self.action_horiozon["open_speed"] *
                                delta_time / 180 * torch.pi)).to(self.device)

            target_ee_pose[:,
                           3:7] = math_utils.quat_mul(delta_quat,
                                                      target_ee_pose[:, 3:7])

        self.target_ee_traj = torch.cat([self.target_ee_traj, target_ee_pose],
                                        dim=0)

        return target_ee_pose

    def success_or_not(self, observation):

        open_or_not = abs(
            self.kitchen._data.joint_pos[0][self.joint_ids]) > self.open_offset

        return open_or_not
