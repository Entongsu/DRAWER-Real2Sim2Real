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
from tools.curobo_planner import IKPlanner, MotionPlanner


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg)


def main():
    """Random actions agent with Isaac Lab environment."""
    save_config, config = save_params_to_yaml(args_cli)
    # create environment configuration
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # joint_pose = torch.as_tensor(real_jps[0]).to(env.device)
    # current_joint_pose = torch.cat(
    #     [joint_pose.unsqueeze(0),
    #      torch.ones(1, 2).to(env.device)], dim=1)
    # for i in range(10):
    #     env.scene["robot"].root_physx_view.set_dof_positions(
    #         torch.as_tensor(current_joint_pose).to(env.device),
    #         indices=torch.arange(env.num_envs).to(env.device))
    #     action_joint_pose = torch.cat(
    #         [joint_pose.unsqueeze(0),
    #          torch.ones(1, 1).to(env.device)], dim=1)

    #     observation, reward, terminate, time_out, info = env.step(
    #         action_joint_pose)
    while True:
        print(env.episode_length_buf)

        # joint_pose = torch.as_tensor(real_jps[i]).to(env.device)
        # current_joint_pose = torch.cat(
        #     [joint_pose.unsqueeze(0),
        #      torch.ones(1, 2).to(env.device)], dim=1)
        current_joint_pose = torch.as_tensor([[
            3.52359504e-01, 3.28257352e-01, 2.84317881e-01, -2.29450774e+00,
            -2.47353196e+00, 1.98522544e+00, -1.15388060e+00, 1.0, 1.0
        ]])

        env.scene["robot"].root_physx_view.set_dof_positions(
            current_joint_pose.to(env.device),
            indices=torch.arange(env.num_envs).to(env.device))

        observation, reward, terminate, time_out, info = env.step(
            torch.zeros((1, 7)).to(env.device))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
