from isaaclab.managers import RewardTermCfg as RewTerm
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject

import isaaclab.utils.math as math_utils
import omni.usd
from tools.curobo_planner import get_mesh_attrs
from curobo.util.usd_helper import UsdHelper

from curobo.geom.types import (
    Mesh,
    WorldConfig,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import numpy as np
from source.isaaclab.isaaclab.envs.mdp.rewards import *
from scripts.workflows.open_policy.utils.criterion import criterion_pick_place


class RewardObsBuffer:

    def __init__(self,
                 target_reward=None,
                 target_obs=None,
                 env_cfg=None,
                 target_object_name=None,
                 placement_object_name=None,
                 ee_frame_name=None,
                 left_finger_name=None,
                 right_finger_name=None,
                 use_gripper_offset=True):
        self.reward_target = target_reward
        self.obs_target = target_obs
        self.env_cfg = env_cfg
        self.target_object_name = target_object_name
        self.placement_object_name = placement_object_name
        self.ee_frame_name = ee_frame_name
        self.left_finger_name = left_finger_name
        self.right_finger_name = right_finger_name
        self.use_gripper_offset = use_gripper_offset

        self.init_ee_pose = torch.as_tensor(
            env_cfg["params"]["Task"]["init_ee_pose"]).unsqueeze(0)

        self.init_ee_object_dist = 0.0011
        self.init_object_height = 0.0011
        self.init_ee_object_angle = 0.0001
        self.init_object_pos = None
        self.minimal_height = self.env_cfg["params"]["Task"]["lift_height"]
        if target_reward is not None:
            self.init_setting()
        self.init_mesh = False

    def init_object_mesh(self, env):

        target_object = env.scene[self.target_object_name]

        target_mesh_name = self.env_cfg["params"]["Task"]["target_mesh_name"]

        prim_path = target_object.cfg.prim_path.replace(".*", str(0))
        target_prim = env.scene.stage.GetPrimAtPath(prim_path)
        mesh_pose = target_prim.GetAttribute("xformOp:translate").Get()
        mesh_orientation = target_prim.GetAttribute("xformOp:orient").Get()
        target_prim.GetAttribute("xformOp:scale").Get()
        mesh_pose_tensor = torch.tensor(
            [[mesh_pose[0], mesh_pose[1], mesh_pose[2]]], dtype=torch.float32)

        # Convert mesh_orientation (Quatd) to tensor
        mesh_orientation_tensor = torch.tensor(
            [[mesh_orientation.GetReal(), *mesh_orientation.GetImaginary()]],
            dtype=torch.float32)

        target_mesh_prim = env.scene.stage.GetPrimAtPath(prim_path + "/" +
                                                         target_mesh_name)

        mesh = UsdGeom.Mesh(target_mesh_prim)
        points = mesh.GetPointsAttr().Get()

        scale = torch.as_tensor(self.env_cfg["params"]["RigidObject"][
            self.target_object_name]["scale"]).to(env.device)

        self.object_vertices = (
            torch.as_tensor(np.array(points).reshape(-1, 3)).to(env.device) *
            scale[0])
        self.object_bbox = torch.cat([
            torch.min(self.object_vertices, dim=0).values,
            torch.max(self.object_vertices, dim=0).values
        ],
                                     dim=0)

        self.object_bbox = self.object_bbox.reshape(
            2, 3).unsqueeze(0).repeat_interleave(env.num_envs, 0)
        self.target_object_type = self.env_cfg["params"]["Task"][
            "target_object_type"]

    def init_setting(self):
        self.success_condition = self.env_cfg["params"]["Task"][
            "success_condition"]

        setattr(
            self.reward_target, "object_ee_distance",
            RewTerm(func=self.object_ee_distance,
                    params={
                        "object_name": self.target_object_name,
                        "ee_frame_name": self.ee_frame_name
                    },
                    weight=0.1))
        setattr(
            self.reward_target, "object_is_lifted",
            RewTerm(
                func=self.object_is_lifted,
                weight=1.0,
                params={
                    "object_name": self.target_object_name,
                },
            ))

        setattr(
            self.reward_target, "object_in_between_gripper",
            RewTerm(
                func=self.object_in_between_gripper,
                weight=1,
                params={
                    "object_name": self.target_object_name,
                },
            ))

        setattr(
            self.reward_target, "object_inside_place_region",
            RewTerm(
                func=self.object_inside_place_region,
                weight=0.8,
                params={
                    "object_name": self.target_object_name,
                },
            ))

        setattr(self.reward_target, "robot_pose_penalty",
                RewTerm(
                    func=self.robot_pose_penalty,
                    weight=0.1,
                ))

        from isaaclab.managers import ObservationTermCfg as ObsTerm
        setattr(
            self.obs_target, "target_object_pos",
            ObsTerm(
                func=self.target_object_pos,
                params={
                    "object_name": self.placement_object_name,
                },
            ))

    def robot_pose_penalty(self, env):

        joint_pos_limits_penalty = joint_pos_limits(env)
        action_rate_l2_penalty = action_rate_l2(env)
        joint_vel_l2_penalty = joint_vel_l2(env)

        return -(joint_pos_limits_penalty * 5 + action_rate_l2_penalty +
                 joint_vel_l2_penalty)

    def object_inside_place_region(self, env, object_name: str):
        if env.episode_length_buf[
                0] < self.env_cfg["params"]["Task"]["reset_horizon"] - 1:

            return torch.zeros(env.num_envs).to(env.device)

        object_state = self.get_root_link_state(env, object_name)
        object_state[:, :3] -= env.scene.env_origins

        placement_state = self.get_root_link_state(env,
                                                   self.placement_object_name)
        placement_state[:, :3] -= env.scene.env_origins
        object_inside_place_region = (
            (object_state[:, 1] > placement_state[:, 1] -
             self.success_condition["bbox_region"][1]) &
            (object_state[:, 1] < placement_state[:, 1] +
             self.success_condition["bbox_region"][1]) &
            (object_state[:, 0] > placement_state[:, 0] -
             self.success_condition["bbox_region"][0]) &
            (object_state[:, 0] < placement_state[:, 0] +
             self.success_condition["bbox_region"][0]))

        # object_not_inside_place_region = ~object_inside_place_region

        reach_target, _ = criterion_pick_place(
            env, object_name, self.placement_object_name,
            self.success_condition["bbox_region"])
        gripper_actions = env.action_manager.get_term(
            "gripper_action").raw_actions.T

        reward_condition = torch.logical_and(
            object_inside_place_region,
            torch.logical_and(gripper_actions[0] >= 0.0, reach_target))
        return torch.where(reward_condition, 15,
                           0) / env.step_dt + torch.where(
                               object_inside_place_region, 3, 0) / env.step_dt

    def target_object_pos(self, env, object_name):

        object_root_pose = env.scene[object_name].data.root_link_state_w.clone(
        )
        object_root_pose[:, :3] -= env.scene.env_origins

        return object_root_pose[:, :3]

    def reset(self, env, object_name, ee_frame_name):

        object = env.scene[object_name]

        ee_pos_w = env.scene[ee_frame_name].data.root_link_state_w.clone()
        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_link_state_w.clone()

        self.init_ee_object_dist = torch.norm(object_pos_w[:, :3] -
                                              ee_pos_w[:, :3],
                                              dim=1)

    def sum_rewards(self, env, object_name, ee_frame_name):

        reward_object_ee_distance = self.object_ee_distance(
            env, object_name, ee_frame_name)
        reward_object_is_lifted = self.object_is_lifted(env, object_name)
        reward_object_in_between_gripper = self.object_in_between_gripper(
            env, object_name)

        return (reward_object_ee_distance + reward_object_is_lifted +
                reward_object_in_between_gripper) * env.step_dt

    def get_root_link_state(self, env, object_name):
        object = env.scene[object_name]
        if object_name == self.placement_object_name:
            object_pose = object.data.root_link_state_w.clone()
            object_pose[:, 2] += 0.25

            return object_pose

        else:
            return object.data.root_link_state_w.clone()

    def add_ee_offset(self, env, ee_pos_w):
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_w[:, :3], ee_pos_w[:, 3:7],
            torch.tensor([[0.0000, 0.0000, 0.1070]],
                         device=env.device).repeat_interleave(env.num_envs, 0),
            torch.tensor([[1., 0., 0., 0.]],
                         device=env.device).repeat_interleave(env.num_envs, 0))
        ee_pos_w = torch.cat([ee_pose_b, ee_quat_b], dim=1)
        return ee_pos_w

    def object_ee_distance(self,
                           env: ManagerBasedRLEnv,
                           object_name: str,
                           ee_frame_name: str,
                           use_gripper_offset=True):

        if not self.init_mesh:
            self.init_object_mesh(env)
            self.init_mesh = True

        if env.episode_length_buf[
                0] < self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.finger_sample = torch.bernoulli(
                torch.full((env.num_envs, ), 0.5)).to(torch.int)

            return torch.zeros(env.num_envs).to(env.device)
        """Reward the agent for reaching the object using tanh-kernel."""

        # reset the distance when the episode is reset

        object_pos_w = self.get_root_link_state(env, object_name)
        ee_pos_w = self.get_root_link_state(env, ee_frame_name)

        if self.use_gripper_offset:
            ee_pos_w = self.add_ee_offset(env, ee_pos_w)

        # calculate the angle between the object and the end-effector
        delta_ee_quat = math_utils.quat_mul(
            math_utils.quat_inv(self.init_ee_pose[:, 3:7].repeat_interleave(
                env.num_envs, 0).to(env.device)), ee_pos_w[:, 3:7])
        delta_ee_axis_angles = math_utils.axis_angle_from_quat(delta_ee_quat)

        delta_object_axis_angles = math_utils.axis_angle_from_quat(
            object_pos_w[:, 3:7])

        delta_ee_object_z_angle = delta_ee_axis_angles[:,
                                                       2] - delta_object_axis_angles[:,
                                                                                     2]

        # Distance of the end-effector to the object: (num_envs,)
        if self.target_object_type == "hollow":
            left_finger_name_state = self.get_root_link_state(
                env, self.left_finger_name)
            right_finger_name_state = self.get_root_link_state(
                env, self.right_finger_name)
            if self.use_gripper_offset:
                left_finger_name_state = self.add_ee_offset(
                    env, left_finger_name_state)
                right_finger_name_state = self.add_ee_offset(
                    env, right_finger_name_state)
            finger_state = torch.cat([
                left_finger_name_state[:, None], right_finger_name_state[:,
                                                                         None]
            ],
                                     dim=1)

            batch_indices = torch.arange(finger_state.shape[0],
                                         device=finger_state.device)

            final_finger_state = finger_state[batch_indices,
                                              self.finger_sample]

            object_ee_distance = torch.norm(object_pos_w[:, :3] -
                                            final_finger_state[:, :3],
                                            dim=1)

        else:
            object_ee_distance = torch.norm(object_pos_w[:, :3] -
                                            ee_pos_w[:, :3],
                                            dim=1)

        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_ee_object_dist = object_ee_distance.clone() + 0.05
            self.init_ee_object_angle = delta_ee_object_z_angle.clone()

        # Define factors for the distance ratio
        factor_high = 1.0  # Factor when distance ratio is greater than 0.7
        factor_low = 1.0  # Factor when 0 <= distance ratio <= 0.7
        factor_negative = 0.5  # Factor when distance ratio < 0

        # Compute the distance ratio
        distance_ratio = 1 - object_ee_distance / self.init_ee_object_dist

        # Apply different factors based on the distance ratio
        scaling_factor = torch.where(
            distance_ratio < 0, factor_negative,
            torch.where(distance_ratio > 0.7, factor_high, factor_low))
        object_ee_dist_reward = torch.clip(
            (1 / torch.clip(object_ee_distance, 0.01, 10) -
             1 / self.init_ee_object_dist) * scaling_factor, -0.0, 3)

        # Compute the reward with the appropriate scaling factor
        # object_ee_dist_reward = torch.clip(distance_ratio, -0.2,
        #                                    1.0) * scaling_factor

        object_ee_angle_penalty = torch.clip(abs(delta_ee_object_z_angle), 0,
                                             1) * 0.1
        object_xyz_angle_penalty = torch.clip(
            torch.sum(abs(delta_ee_axis_angles[:, :2]), dim=1) - 0.4, 0.0,
            3) * 0.1

        # x angle can not move a lot

        ee_xy_angles_penalty = torch.sum(abs(delta_ee_axis_angles[:, :2]),
                                         dim=1)

        rewards = (object_ee_dist_reward - object_ee_angle_penalty -
                   ee_xy_angles_penalty -
                   object_xyz_angle_penalty) * 1 / env.step_dt

        return rewards

    def object_is_lifted(self, env, object_name: str):

        pick_object_state = self.get_root_link_state(env, object_name)
        target_pose = self.get_root_link_state(env, self.placement_object_name)
        target_pose[:, :3] -= env.scene.env_origins

        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_object_height = pick_object_state[:, 2].clone()

            self.init_object_to_lift_dist = torch.linalg.norm(
                pick_object_state[:, :3] - target_pose[:, :3] -
                env.scene.env_origins,
                dim=1)

        elif env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)

        cur_object_to_lift_dist = torch.linalg.norm(
            pick_object_state[:, :3] - target_pose[:, :3] -
            env.scene.env_origins[:, :3],
            dim=1)

        target_reward = torch.clip(
            (self.init_object_to_lift_dist - cur_object_to_lift_dist) /
            self.init_object_to_lift_dist, -0.0, 1) * 5

        delta_height = torch.clip(
            (pick_object_state[:, 2]) - self.init_object_height, 0.0, 0.2) * 50
        negative_delta_height_index = torch.where(
            (pick_object_state[:, 2] - self.init_object_height - 0.20) > 0.0,
            True, False)
        delta_height[negative_delta_height_index] = -torch.clip(
            ((pick_object_state[:, 2] - self.init_object_height -
              0.40))[negative_delta_height_index], 0.0, 0.2) * 10

        bonus_reward = torch.where(
            (pick_object_state[:, 2] - self.init_object_height) > 0.01,
            torch.ones(env.num_envs).to(env.device) * 1,
            torch.zeros(env.num_envs).to(env.device) * 0.0)

        return target_reward / env.step_dt + bonus_reward / env.step_dt + delta_height / env.step_dt

    def object_in_between_gripper(self,
                                  env,
                                  object_name: str,
                                  use_gripper_offset=True):
        left_finger_name_state = self.get_root_link_state(
            env, self.left_finger_name)
        right_finger_name_state = self.get_root_link_state(
            env, self.right_finger_name)
        ee_pos_w = self.get_root_link_state(env, self.ee_frame_name)
        object_state = self.get_root_link_state(env, object_name)

        if self.use_gripper_offset:
            ee_pos_w = self.add_ee_offset(env, ee_pos_w)

        height_diff = object_state[:, 2] - ee_pos_w[:, 2]

        height_enough = torch.where(abs(height_diff) < 0.04, True, False)
        object_pos_w = self.get_root_link_state(env, object_name)
        if self.target_object_type == "hollow":

            left_finger_name_state = self.get_root_link_state(
                env, self.left_finger_name)
            right_finger_name_state = self.get_root_link_state(
                env, self.right_finger_name)
            if self.use_gripper_offset:
                left_finger_name_state = self.add_ee_offset(
                    env, left_finger_name_state)
                right_finger_name_state = self.add_ee_offset(
                    env, right_finger_name_state)
            finger_state = torch.cat([
                left_finger_name_state[:, None], right_finger_name_state[:,
                                                                         None]
            ],
                                     dim=1)

            batch_indices = torch.arange(finger_state.shape[0],
                                         device=finger_state.device)

            final_finger_state = finger_state[batch_indices,
                                              self.finger_sample]

            object_ee_distance = torch.norm(object_pos_w[:, :3] -
                                            final_finger_state[:, :3],
                                            dim=1)

            close_enough = torch.where(object_ee_distance < 0.06, True, False)

            cur_object_bbox = math_utils.transform_points(
                self.object_bbox.clone(), object_pos_w[:, :3],
                object_pos_w[:, 3:7])
            cur_object_bbox[:, 0, :] += 0.01
            cur_object_bbox[:, 1, :] -= 0.01
            in_between_gripper_left = torch.logical_and(
                torch.logical_and(
                    cur_object_bbox[:, 0, 0] < left_finger_name_state[:, 0],
                    cur_object_bbox[:, 1, 0] > left_finger_name_state[:, 0]),
                torch.logical_and(
                    cur_object_bbox[:, 0, 1] < left_finger_name_state[:, 1],
                    cur_object_bbox[:, 1, 1] > left_finger_name_state[:, 1]))

            in_between_gripper_right = torch.logical_and(
                torch.logical_and(
                    cur_object_bbox[:, 0, 0] < right_finger_name_state[:, 0],
                    cur_object_bbox[:, 1, 0] > right_finger_name_state[:, 0]),
                torch.logical_and(
                    cur_object_bbox[:, 0, 1] < right_finger_name_state[:, 1],
                    cur_object_bbox[:, 1, 1] > right_finger_name_state[:, 1]))
            in_between_gripper = torch.logical_or(in_between_gripper_right,
                                                  in_between_gripper_left)

        else:
            in_between_gripper_y = torch.logical_or(
                torch.logical_and(
                    object_state[:, 1] > left_finger_name_state[:, 1],
                    object_state[:, 1] < right_finger_name_state[:, 1]),
                torch.logical_and(
                    object_state[:, 1] < left_finger_name_state[:, 1],
                    object_state[:, 1] > right_finger_name_state[:, 1]))

            in_between_gripper_x = torch.logical_or(
                torch.logical_and(
                    object_state[:, 0] > left_finger_name_state[:, 0],
                    object_state[:, 0] < right_finger_name_state[:, 0]),
                torch.logical_and(
                    object_state[:, 0] < left_finger_name_state[:, 0],
                    object_state[:, 0] > right_finger_name_state[:, 0]))
            in_between_gripper = torch.logical_or(in_between_gripper_x,
                                                  in_between_gripper_y)

            object_ee_distance = torch.norm(object_state[:, :2] -
                                            ee_pos_w[:, :2],
                                            dim=1)

            close_enough = torch.where(object_ee_distance < 0.05, True, False)

        in_between_gripper = torch.logical_and(
            in_between_gripper,
            close_enough,
        )
        in_between_gripper = torch.logical_and(in_between_gripper,
                                               height_enough)

        if env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)

        # penalty of the gripper close(-1) when the object is not in between

        ation_penalty = self.gripper_in_between_penalty(
            env, in_between_gripper)

        in_between_reward = torch.where(
            in_between_gripper[None],
            torch.ones(env.num_envs).to(env.device) * 0.5,
            -torch.ones(env.num_envs).to(env.device) * 0.0) / env.step_dt

        return (in_between_reward + ation_penalty)[0]

    def gripper_in_between_penalty(self, env, in_between_gripper):

        # Check whether the gripper is in-between
        gripper_condition = in_between_gripper[None]

        # Define actions (assumes gripper action is the relevant term)
        actions = env.action_manager.get_term("gripper_action").raw_actions.T

        # Apply rewards/penalties based on the condition
        action_penalty = torch.where(
            gripper_condition,
            # If in-between: open = penalty, close = reward
            torch.clip(-2 * torch.sign(actions + 0.02), -0.0, 1) * 1.0,
            # If not in-between: open = reward, close = penalty
            torch.clip(2 * torch.sign(actions + 0.02), -0.0, 0.0) *
            2) / env.step_dt

        return action_penalty
