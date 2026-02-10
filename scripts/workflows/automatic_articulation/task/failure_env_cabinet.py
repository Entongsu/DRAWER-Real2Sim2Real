import random

import torch
import copy
import numpy as np
import isaaclab.utils.math as math_utils


class FailureEnvCabinet:

    def __init__(self, env, failure_type):
        self.multi_env = env
        self.failure_type = failure_type
        self.planner = self.multi_env.planner

        self.reach_cabinet_horizon = self.multi_env.curobo_planner_length
        self.approach_cabinet_horizon = self.multi_env.env_cabinet.approach_horizon
        self.close_gripper_horizon = self.multi_env.env_cabinet.close_gripper_horizon
        self.open_cabinet_horizon = self.multi_env.env_cabinet.open_cabinet_horizon

        self.start_frame = self.reach_cabinet_horizon + self.approach_cabinet_horizon + self.close_gripper_horizon + self.open_cabinet_horizon

        self.init_setting()

    def init_setting(self):

        self.xyz_grasp_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["xyz_grasp_offset"]
        self.slip_frame_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["slip_frame_offset"]
        self.rpy_grasp_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["rpy_grasp_offset"]
        self.open_speed_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["open_speed_offset"]
        self.xyz_grasp_sign_offset = self.multi_env.env_config["params"][
            "Task"]["Failure"]["grasper"]["xyz_grasp_sign_offset"]

    def miss_placement_failure_env(self, demo=None, last_obs=None):
        pass

    def miss_grasp_failure_env(self, demo=None, last_obs=None):

        action = copy.deepcopy(np.asarray(demo["actions"]))
        grasp_action = torch.as_tensor(action).to(
            self.multi_env.device)[:self.reach_object_horizon +
                                   self.approach_cabinet_horizon +
                                   self.close_gripper_horizon +
                                   self.lift_object_horizon]
        self.multi_env.env_placement.get_target_placement_traj(
            grasp_action[-1])
        placement_action = self.multi_env.env_placement.target_ee_traj

        remainning_step = self.total_frame - placement_action.shape[0]
        remainning_action = placement_action[-1].unsqueeze(0).repeat(
            remainning_step, 1)
        failure_reasoning = {self.failure_type: "miss_grasp"}
        return torch.cat([placement_action,
                          remainning_action]), failure_reasoning

    def mistake_pickup_place_failure_env(self, demo=None, last_obs=None):
        pass

    def xyz_offset_failure_env(
        self,
        demo=None,
        last_obs=None,
    ):

        action = copy.deepcopy(np.asarray(demo["actions"]))

        random_xyz_offset = [
            random.uniform(*self.xyz_grasp_offset[i]) *
            random.choice(self.xyz_grasp_sign_offset[i])
            if f"{dim}_offset" in self.failure_type else 0
            for i, dim in enumerate(["x", "y", "z"])
        ]

        random_rpy_offset = [
            random.uniform(*self.rpy_grasp_offset[i]) * random.choice([-1, 1])
            if f"{dim}_offset" in self.failure_type else 0
            for i, dim in enumerate(["roll", "pitch", "yaw"])
        ]

        target_pose = action[self.reach_cabinet_horizon]
        refine_pose = action[self.reach_cabinet_horizon +
                             self.approach_cabinet_horizon]

        target_pose = torch.as_tensor(target_pose).to(self.multi_env.device)
        refine_pose = torch.as_tensor(refine_pose).to(self.multi_env.device)
        random_rpy_offset = torch.as_tensor(random_rpy_offset).to(
            self.multi_env.device)
        random_xyz_offset = torch.as_tensor(random_xyz_offset).to(
            self.multi_env.device)
        target_pose[:3] += random_xyz_offset
        refine_pose[:3] += random_xyz_offset

        if any(random_rpy_offset):

            orientation_offset = math_utils.quat_from_euler_xyz(
                random_rpy_offset[0], random_rpy_offset[1],
                random_rpy_offset[2]).to(self.multi_env.device)
            target_pose[3:7] = math_utils.shortest_angles(
                math_utils.quat_mul(orientation_offset,
                                    target_pose[3:7]).unsqueeze(0),
                self.multi_env.env)[0]
            refine_pose[3:7] = target_pose[3:7].clone()

        obs = demo["obs"]
        self.multi_env.env_cabinet.robot_pose = refine_pose[..., :7]

        self.multi_env.env_cabinet.get_open_dir()
        self.multi_env.env_cabinet.open_dir += random_rpy_offset[2]
        self.multi_env.open_dir = math_utils.axis_angle_from_quat(
            target_pose[3:7])
        self.multi_env.env_cabinet.plan_traj(
            torch.as_tensor(np.array([obs["joint_pos"][0]])).to(
                self.multi_env.device).clone()[..., :8], target_pose)

        sign = "+" if any(
            [v > 0 for v in random_xyz_offset + random_rpy_offset]) else "-"
        failure_reasoning_template = f"The robot failed to grasp the handle due to the {{}} offset."
        failure_reasoning = failure_reasoning_template.format(
            f"{sign} {self.failure_type.split('_')[0]} offset")

        return self.multi_env.env_cabinet.target_ee_traj, failure_reasoning

        # return grasp_action

    def slip_failure_env(self, demo=None, last_obs=None):

        action = copy.deepcopy(np.asarray(demo["actions"]))

        start_action_index = int(self.slip_frame_offset[0])

        end_action_index = int(self.slip_frame_offset[1])

        random_open_gripper_frame = random.randint(start_action_index,
                                                   end_action_index)

        action[random_open_gripper_frame:, -1] = 1
        random_bool = bool(random.getrandbits(1))
        failure_reasoning = {self.failure_type: "slip"}

        return torch.as_tensor(action).to(
            self.multi_env.device), failure_reasoning

    def check_success(self, object, last_obs):

        return self.multi_env.env_cabinet.success_or_not(last_obs)

    def episode_check_success(self, last_obs):

        return self.multi_env.env_cabinet.success_or_not(last_obs)
