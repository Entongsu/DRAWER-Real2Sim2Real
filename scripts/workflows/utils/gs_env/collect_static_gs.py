# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

sys.path.append(".")

from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

import imageio
from isaaclab.utils import Timer
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--disable_fabric",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",
                    type=int,
                    default=None,
                    help="Number of environments to simulate.")

parser.add_argument("--filename",
                    type=str,
                    default=None,
                    help="h5py dataset path")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from tools.deformable_obs import *
from scripts.workflows.sysID.utils.attachment_utils import GripperActionManager, hide_prim

from scripts.workflows.sysID.utils.data_utilis import DataBuffer
from scripts.workflows.sysID.ASID.tool.utilis import save_target_video
from tools.data_collection import load_dataset_saver


def setup_env(args_cli):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    return gym.make(args_cli.task, cfg=env_cfg)


def initialize_gripper(env,
                       args_cli,
                       sample_id=None,
                       static_frames=3,
                       gripper_offset=0.0,
                       gripper_offset_xyz=None):

    gripper_manager = GripperActionManager(
        env,
        args_cli.task,
        ee_height_compensate=0.0486,
        static_frames=static_frames,
        sample_seg_id=sample_id,
        gripper_offset=gripper_offset,
        gripper_offset_xyz=gripper_offset_xyz)

    if "Abs" in args_cli.task or "Rel" in args_cli.task:
        for i in range(env.num_envs):
            hide_prim(env.scene.stage, f"/World/envs/env_{i}/gripper")
    return gripper_manager


def initialize_buffers(env, obs, device, exclude_obs_keys=[], sample_id=None):
    obs_keys = [*obs[0]["policy"].keys()]

    # remove the whole point cloud and segmentation point cloud
    for name in exclude_obs_keys:
        if name in obs_keys:
            obs_keys.remove(name)

    return DataBuffer(env, device, obs_keys, sample_id=sample_id)


def create_log_dirs(base_dir="logs/rabbit/static_gs/"):
    os.makedirs(base_dir + "/video", exist_ok=True)
    return base_dir


def reset_obs_cam(env, names, bool=False):
    for name in names:
        env.scene["deform_object"].cfg.deform_cfg["camera_obs"][name] = bool


def on_step(gripper_manager, target_gripper_traj, gripper_actions, next_obs):

    if "Abs" in args_cli.task or "Rel" in args_cli.task:
        actions = gripper_manager.assemble_actions(
            target_gripper_traj, gripper_actions,
            next_obs["policy"]["ee_pose"])
    else:
        actions = gripper_manager.assemble_actions(target_gripper_traj,
                                                   gripper_actions, None)
    return actions


def trigger_reset(
    gripper_manager,
    target_gripper_traj,
    gripper_actions,
):

    if "Abs" in args_cli.task or "Rel" in args_cli.task:

        next_obs = gripper_manager.technical_reset(
            target_gripper_traj,
            gripper_actions,
        )
    else:
        next_obs = gripper_manager.technical_reset()
    return next_obs


def sample_grasp_index(gripper_manager, num_grasp=100):

    sample_explore_grasp_indices = gripper_manager.sample_grasp_index_from_pc(
        num_grasp)

    # Randomly select 10 values from the tensor
    random_indices = torch.randperm(sample_explore_grasp_indices.size(0))
    sequeece_grasp_index = sample_explore_grasp_indices[random_indices]
    return sequeece_grasp_index


def main():
    env = setup_env(args_cli)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()
    log_dir = create_log_dirs()

    random_orientation_range = [0, 3.14]
    proposed_position_dir = torch.as_tensor([0.0, 0.0, 0.01]).to(env.device)
    gripper_actions = torch.as_tensor([1]).to(env.device)
    gripper_offset_xyz = torch.as_tensor(
        env.scene["deform_object"].cfg.deform_cfg["gripper_offset_xyz"]).to(
            env.device).repeat(env.num_envs, 1)
    sample_id = (torch.arange(env.num_envs) + 1).to(env.device) * env.scene[
        "deform_object"].cfg.deform_cfg["camera_obs"]["segmentation_id"]

    gripper_manager = initialize_gripper(
        env,
        args_cli,
        sample_id,
        gripper_offset=env.scene["deform_object"].cfg.
        deform_cfg["gripper_offset"] + 0.01,
        gripper_offset_xyz=gripper_offset_xyz)
    buffer = initialize_buffers(env,
                                obs,
                                device="cuda:0",
                                exclude_obs_keys=["whole_pc", "seg_pc"],
                                sample_id=sample_id)

    while simulation_app.is_running():
        for loop_index in range(1):
            buffer.clear_buffer("target")
            collector = load_dataset_saver(args_cli, log_dir, loop_index)

            for _ in range(env.max_episode_length):
                buffer.clear_buffer("target")
                transition = {}

                if env.episode_length_buf[0] == 0:

                    random_grasp_index = sample_grasp_index(
                        gripper_manager, num_grasp=env.num_envs)

                    proposed_orientation = gripper_manager.random_orientation(
                        random_orientation_range) * 0

                    reset_obs_cam(env, names=["seg_pc", "whole_pc"], bool=True)

                    target_gripper_traj = gripper_manager.on_training_start(
                        proposed_orientation,
                        proposed_position_dir=proposed_position_dir * 0,
                        sample_grasp_indices=random_grasp_index)
                    next_obs = gripper_manager.technical_reset(
                        target_gripper_traj, gripper_actions)

                    reset_obs_cam(env,
                                  names=["seg_pc", "whole_pc"],
                                  bool=False)

                    # trigger reset for the gripper
                    next_obs = trigger_reset(gripper_manager,
                                             target_gripper_traj,
                                             gripper_actions)

                # generate next actions
                actions = on_step(gripper_manager, target_gripper_traj,
                                  gripper_actions, next_obs)
                next_obs, reward, terminate, time_out, info = env.step(actions)

                if env.episode_length_buf[
                        0] % 5 == 0 and env.episode_length_buf[0] > 5:
                    randomize_camera_pose(env, torch.arange(env.num_envs),
                                          env.episode_length_buf * 0.4)

                    transition.update({
                        "obs": next_obs["policy"],
                        "reward": reward * 0,
                        "action": actions
                    })

                    buffer.cache_traj(transition, cache_type="target")

            buffer.store_transition("target")
            buffer.clear_cache("target")

            for key in buffer.target_buffer:
                if key != "actions":
                    collector.add(f"obs/{key}", buffer.target_buffer[key])
            collector.add("actions",
                          buffer.target_buffer["actions"].cpu().numpy())
            collector.flush(
                np.arange(len(buffer.target_buffer["actions"])).astype(
                    np.int16))

            collector.close()

        break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
