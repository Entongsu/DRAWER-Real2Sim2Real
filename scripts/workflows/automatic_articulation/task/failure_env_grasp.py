import random

import torch
import copy
import numpy as np
import isaaclab.utils.math as math_utils


class FailureEnvGrasp:

    def __init__(self, env, failure_type):
        self.multi_env = env
        self.failure_type = failure_type
        self.planner = self.multi_env.planner

        self.reach_object_horizon = self.multi_env.curobo_planner_length
        self.approach_object_horizon = self.multi_env.env_grasp.approach_horizon
        self.close_gripper_horizon = self.multi_env.env_grasp.close_gripper_horizon
        self.lift_object_horizon = self.multi_env.env_grasp.lift_object_horizon
        self.reach_cabinet_horizon = self.multi_env.curobo_planner_length
        self.release_object_horizon = self.multi_env.env_placement.open_gripper_horizon
        try:
            self.total_frame = self.reach_object_horizon + self.approach_object_horizon + self.close_gripper_horizon + self.lift_object_horizon + self.reach_cabinet_horizon + self.release_object_horizon + self.multi_env.env_placement.refine_horizon
        except:
            self.total_frame = self.reach_object_horizon + self.approach_object_horizon + self.close_gripper_horizon + self.lift_object_horizon + self.reach_cabinet_horizon + self.release_object_horizon
        self.init_setting()

    def init_setting(self):

        self.xyz_grasp_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["xyz_grasp_offset"]
        self.xyz_placement_offset = self.multi_env.env_config["params"][
            "Task"]["Failure"]["grasper"]["xyz_placement_offset"]
        self.slip_frame_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["slip_frame_offset"]
        self.rpy_grasp_offset = self.multi_env.env_config["params"]["Task"][
            "Failure"]["grasper"]["rpy_grasp_offset"]

    def miss_placement_failure_env(self, demo=None, last_obs=None):
        action = copy.deepcopy(np.asarray(demo["actions"]))
        grasp_action = torch.as_tensor(action).to(
            self.multi_env.device)[:self.reach_object_horizon +
                                   self.approach_object_horizon +
                                   self.close_gripper_horizon +
                                   self.lift_object_horizon]

        remainning_step = self.total_frame - grasp_action.shape[0]
        remainning_action = grasp_action[-1].unsqueeze(0).repeat(
            remainning_step, 1)
        failure_reasoning = {self.failure_type: "miss_placement"}
        return torch.cat([grasp_action, remainning_action]), failure_reasoning

    def miss_grasp_failure_env(self, demo=None, last_obs=None):

        action = copy.deepcopy(np.asarray(demo["actions"]))
        grasp_action = torch.as_tensor(action).to(
            self.multi_env.device)[:self.reach_object_horizon +
                                   self.approach_object_horizon +
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
        keys = list(self.multi_env.env.scene.rigid_objects.keys())
        target_object_name, placement_object_name = self.multi_env.target_object_name, self.multi_env.placement_object_name

        keys.remove(target_object_name)
        keys.remove(placement_object_name)

        if self.failure_type == "mistake_pickup":
            new_target_object_name = random.choice(keys)
            new_placement_object_name = placement_object_name
        if self.failure_type == "mistake_place":
            new_placement_object_name = random.choice(keys)
            new_target_object_name = target_object_name
        if self.failure_type == "mistake_pickup_place":
            # remove the correct pickup object
            keys = list(self.multi_env.env.scene.rigid_objects.keys())
            keys.remove(target_object_name)
            new_target_object_name = random.choice(keys)

            # remove the correct placement object and new target object
            keys = list(self.multi_env.env.scene.rigid_objects.keys())
            keys.remove(new_target_object_name)
            try:
                keys.remove(placement_object_name)
            except:
                pass
            new_placement_object_name = random.choice(keys)

        self.multi_env.assign_objects(new_target_object_name,
                                      new_placement_object_name)

        self.multi_env.env_grasp.reset(last_obs, resample=True)
        grasp_action = self.multi_env.env_grasp.target_ee_traj
        self.multi_env.env_placement.get_target_placement_traj(
            grasp_action[-1])
        placement_action = self.multi_env.env_placement.target_ee_traj
        failure_reasoning = f"Your pick up the {new_target_object_name} and place it onto the {new_placement_object_name}."
        return torch.cat([grasp_action, placement_action]), failure_reasoning

    def xyz_offset_failure_env(
        self,
        demo=None,
        last_obs=None,
    ):

        action = copy.deepcopy(np.asarray(demo["actions"]))
        if "grasp" in self.failure_type:
            mode = "grasp"

            target_pose = action[self.reach_object_horizon - 1]
            refine_pose = action[self.reach_object_horizon +
                                 self.approach_object_horizon +
                                 self.close_gripper_horizon - 1]
        elif "placement" in self.failure_type:
            mode = "placement"
            grasp_action = torch.as_tensor(action).to(
                self.multi_env.device)[:self.reach_object_horizon +
                                       self.approach_object_horizon +
                                       self.close_gripper_horizon +
                                       self.lift_object_horizon]
            target_pose = self.multi_env.env_placement.placement_object._data.root_state_w[
                0, :7].cpu().numpy()
            refine_pose = self.multi_env.env_placement.placement_object._data.root_state_w[
                0, :7].cpu().numpy()

        random_xyz_offset = [
            random.uniform(*self.xyz_placement_offset[i] if mode ==
                           "placement" else self.xyz_grasp_offset[i]) *
            random.choice([-1, 1])
            if f"{dim}_{mode}_offset" in self.failure_type else 0
            for i, dim in enumerate(["x", "y", "z"])
        ]

        random_rpy_offset = [
            random.uniform(*self.rpy_grasp_offset[i]) * random.choice([-1, 1])
            if f"{dim}_{mode}_offset" in self.failure_type else 0
            for i, dim in enumerate(["roll", "pitch", "yaw"])
        ]

        target_pose[:3] += random_xyz_offset
        refine_pose[:3] += random_xyz_offset
        target_pose = torch.as_tensor(target_pose).to(self.multi_env.device)
        refine_pose = torch.as_tensor(refine_pose).to(self.multi_env.device)

        if any(random_rpy_offset):
            orientation_offset = math_utils.quat_from_euler_xyz(
                *random_rpy_offset)
            target_pose[3:7] = math_utils.shortest_angles(
                math_utils.quat_mul(orientation_offset,
                                    target_pose[3:7]).unsqueeze(0),
                self.multi_env.env)[0]

        if mode == "grasp":

            self.multi_env.env_grasp.robot_pose = torch.as_tensor(
                refine_pose[:7]).to(self.multi_env.device)

            grasp_action = self.multi_env.env_grasp.reset_planner_grasp(
                0, target_pose, sample_grasp=False)

            if grasp_action is None:
                return None, None

            self.multi_env.env_placement.get_target_placement_traj(
                grasp_action[-1])
            placement_action = self.multi_env.env_placement.target_ee_traj

        elif mode == "placement":
            result = self.multi_env.env_placement.get_target_placement_traj(
                grasp_action[-1], target_pose)

            if result is None:
                return None, None

            placement_action = self.multi_env.env_placement.target_ee_traj

        sign = "+" if any(
            [v > 0 for v in random_xyz_offset + random_rpy_offset]) else "-"
        failure_reasoning_template = f"The robot failed to {mode} the object due to the {{}} offset."
        failure_reasoning = failure_reasoning_template.format(
            f"{sign} {self.failure_type.split('_')[0]} offset")

        return torch.cat([grasp_action, placement_action]), failure_reasoning

        # return grasp_action

    def slip_failure_env(self, demo=None, last_obs=None):

        action = copy.deepcopy(np.asarray(demo["actions"]))

        start_action_index = int(self.slip_frame_offset[0])
        # self.reach_object_horizon + self.approach_object_horizon + self.close_gripper_horizon + int(
        #     self.slip_frame_offset[0])
        end_action_index = int(self.slip_frame_offset[1])
        # start_action_index + self.lift_object_horizon + int(
        #     self.slip_frame_offset[1])
        random_open_gripper_frame = random.randint(start_action_index,
                                                   end_action_index)

        action[random_open_gripper_frame:, -1] = 1
        random_bool = bool(random.getrandbits(1))
        if random_bool:
            random_close_gripper_frame = random.randint(
                random_open_gripper_frame + 6,
                end_action_index + self.release_object_horizon)
            action[random_close_gripper_frame:, -1] = -1
        failure_reasoning = {self.failure_type: "slip"}
        return torch.as_tensor(action).to(
            self.multi_env.device), failure_reasoning

    def check_success(self, object, last_obs):

        return self.grasp_success and self.multi_env.env_placement.success_or_not(
            last_obs)

    def episode_check_success(self, last_obs):
        if self.grasp_success:
            return
        self.grasp_success = self.multi_env.env_grasp.success_or_not(last_obs)
