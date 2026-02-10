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
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import h5py
import copy

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

from scripts.workflows.automatic_articulation.utils.process_action import normalize_action, unnormalize, reset_env
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg)


def main():
    import sys
    sys.path.append("../../robomimic")
    import robomimic.utils.file_utils as FileUtils
    save_config, config = save_params_to_yaml(args_cli)
    # create environment configuration
    env = setup_env(args_cli, save_config)

    ckpt_path = "../../robomimic/bc_trained_models/diffusion/20241021221407/models/model_epoch_600.pth"
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path,
                                                         device=env.device,
                                                         verbose=True)
    policy.start_episode()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    kitchen = env.scene["kitchen"]
    target_handle_name = kitchen.cfg.articulation_cfg["target_drawer"]
    joint_ids, joint_names = kitchen.find_joints(
        kitchen.cfg.articulation_cfg["robot_random_range"][target_handle_name]
        ["joint_name"])
    robot = env.scene["robot"]

    success_count = 0

    normalize_file = h5py.File(f"{args_cli.log_dir}/{args_cli.filename}.hdf5",
                               'r+')
    if "refine" in args_cli.filename:
        normalize_action = True
        stats = np.load(f"{args_cli.log_dir}/stats.npy",
                        allow_pickle=True).item()
    else:
        normalize_action = False

    for demo_id in range(len(normalize_file["data"].keys())):

        obs = normalize_file["data"][f"demo_{demo_id}"]["obs"]

        ground_actions = normalize_file["data"][f"demo_{demo_id}"]["actions"]

        # reset joint position
        init_joint_pos = torch.as_tensor(obs["joint_pos"][0]).unsqueeze(0).to(
            env.device)

        if args_cli.eval_type == "train" or args_cli.eval_type == "replay":
            last_obs = reset_env(
                env, ground_actions, robot, init_joint_pos,
                torch.as_tensor(obs["robot_base"][0]).unsqueeze(0).to(
                    env.device))
        else:
            last_obs, _ = env.reset()

        # create obs buffer
        obs_buffer = {}
        for key in last_obs["policy"].keys():

            obs_buffer[key] = torch.cat(
                [last_obs["policy"][key], last_obs["policy"][key]], dim=0)

        for actions_id in range(len(ground_actions) + 50):
            # train_obs = {
            #     "ee_pose": obs["ee_pose"][actions_id],
            #     "drawer_pose": obs["drawer_pose"][actions_id],
            #     "joint_pos": obs["joint_pos"][actions_id],
            # }

            for key in last_obs["policy"].keys():
                obs_buffer[key][0] = obs_buffer[key][-1].clone()
                obs_buffer[key][-1] = last_obs["policy"][key]
            # if "drawer_pose" in obs_buffer.keys():
            #     obs_buffer.pop("drawer_pose")

            predicted_action = policy(obs_buffer)
            # predicted_action = policy(train_obs)
            if normalize_action:
                predicted_action = unnormalize(predicted_action,
                                               stats["action"],
                                               scale=100)

                # predicted_action2 = unnormalize(ground_actions[actions_id],
                #                                 stats["action"],
                #                                 scale=100)
                # print(np.max(abs(predicted_action2[:3] -
                #                  predicted_action[:3])))

            predicted_action = torch.as_tensor(predicted_action).unsqueeze(
                0).to(env.device)

            predicted_action[:, -1] = torch.sign(predicted_action[:, -1])

            observation, reward, terminate, time_out, info = env.step(
                predicted_action)

            last_obs = observation

        # save the successful demonstrations
        if env.scene["kitchen"]._data.joint_pos[0][joint_ids] > 0.20:
            success_count += 1

        print("success rate: ", success_count / (demo_id + 1))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
