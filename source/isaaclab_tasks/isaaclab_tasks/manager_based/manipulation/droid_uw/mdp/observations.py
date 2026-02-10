# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ee_pose_w = robot.data.body_state_w[:, 8, :7]
    root_pose_w = robot.data.root_state_w[:, :7]
    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7])
    # # account for the offset
    # ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
    #     ee_pose_b, ee_quat_b,
    #     torch.tensor([[0.0000, 0.0000, 0.1070]], device=robot.device),
    #     torch.tensor([[1., 0., 0., 0.]], device=robot.device))
    gripper = torch.sign(env.scene["robot"]._data.joint_pos[:,
                                                            -1].unsqueeze(0) -
                         0.03)

    return torch.cat([ee_pose_b, ee_quat_b, gripper], dim=1)


def control_joint_action(env: ManagerBasedRLEnv) -> torch.Tensor:

    return env.scene["robot"]._data.joint_pos_target


def kitchen_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    kitchen_data = env.scene["kitchen"].data

    return torch.cat([
        kitchen_data.root_pos_w - env.scene.env_origins,
        kitchen_data.root_quat_w
    ],
                     dim=1)


def kitchen_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    if "kitchen" not in env.scene.keys():
        return {}
    kitchen = env.scene["kitchen"]
    joint_pos = kitchen._data.joint_pos

    return joint_pos


def robot_root_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The root pose of the robot in the environment frame."""
    robot_data: ArticulationData = env.scene["robot"].data

    return torch.cat([
        robot_data.root_pos_w - env.scene.env_origins, robot_data.root_quat_w
    ],
                     dim=1)


def robot_link_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]

    robot_root = robot.data.root_state_w[:, :7]
    link_transforms = robot.data.body_state_w[..., :7]
    num_link = link_transforms.shape[1]
    relative_link_pose = math_utils.subtract_frame_transforms(
        robot_root[:, 0:3].repeat_interleave(num_link, 0),
        robot_root[:, 3:7].repeat_interleave(num_link, 0),
        link_transforms[0, :, 0:3], link_transforms[0, :, 3:7])

    return torch.cat(relative_link_pose, dim=1).unsqueeze(0)


def object_root_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The root pose of the robot in the environment frame."""

    result = {}
    rigdid_objects = env.scene._rigid_objects
    robot_root_pose_w = env.scene["robot"].data.root_state_w[:, :7]
    for object_name in rigdid_objects.keys():
        object_data = env.scene[object_name].data
        object_root_pose_w = object_data.root_state_w[:, :7]

        result[f"{object_name}_root_pose"] = object_root_pose_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            robot_root_pose_w[:, 0:3], robot_root_pose_w[:, 3:7],
            object_root_pose_w[:, 0:3], object_root_pose_w[:, 3:7])
        result[f"{object_name}_pose"] = torch.cat([ee_pose_b, ee_quat_b],
                                                  dim=1)

    return result


def object_pose_relative2robot(env: ManagerBasedRLEnv,
                               object_name) -> torch.Tensor:

    robot_root_base = robot_root_pose(env)

    object_data: ArticulationData = env.scene[object_name].data
    object_root_pose = object_data.root_pos_w - env.scene.env_origins
    translate_robot_to_handle, quat_robot_to_handle = math_utils.subtract_frame_transforms(
        robot_root_base[:, :3], robot_root_base[:, 3:7],
        object_root_pose[:, :3], object_data.root_quat_w)

    return torch.cat([translate_robot_to_handle, quat_robot_to_handle], dim=1)


def drawer_pose_relative2robot(env: ManagerBasedRLEnv, kitchen_name,
                               drawer_name) -> torch.Tensor:

    robot_root_base = robot_root_pose(env)

    kitchen = env.scene[kitchen_name]
    drawer_id, _ = kitchen.find_bodies(drawer_name)

    kitchen_data: ArticulationData = env.scene[kitchen_name].data
    drawer_pose = kitchen_data.body_pos_w[:,
                                          drawer_id[0]] - env.scene.env_origins
    # drawer_pose[:, 0] -= 0.15
    dawer_quat = kitchen_data.body_quat_w[:, drawer_id[0]]
    translate_robot_to_handle, quat_robot_to_handle = math_utils.subtract_frame_transforms(
        robot_root_base[:, :3], robot_root_base[:, 3:7], drawer_pose,
        dawer_quat)

    return torch.cat([translate_robot_to_handle, quat_robot_to_handle], dim=1)


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[...,
                                        0, :] - ee_tf_data.target_pos_w[...,
                                                                        0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[
        ..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv,
            make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
