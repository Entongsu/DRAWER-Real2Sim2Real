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
from scripts.workflows.automatic_articulation.task.env_grasp import GrasperEnv
from scripts.workflows.automatic_articulation.utils.process_action import add_demonstraions_to_buffer
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
import pickle
from curobo.util.usd_helper import UsdHelper
from scripts.workflows.automatic_articulation.utils.grasp_sampler import GraspSampler
from tools.curobo_planner import IKPlanner, MotionPlanner
import yaml


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg)


def main():
    """Random actions agent with Isaac Lab environment."""
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment configuration
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    curobo_ik = IKPlanner(
        env,
        only_paths=[
            "/World/envs/env_0/kitchen",
            "/World/envs/env_0/sugar_box",
        ],
        reference_prim_path="/World/envs/env_0/Robot",
        ignore_substring=[
            "/World/envs/env_0/Robot",
            "/World/GroundPlane",
            "/World/collisions",
            "/World/light",
            "/curobo",
        ],
    )
    # curobo_ik = IKPlanner(env, device=env.device)
    # init_ee_pose = torch.as_tensor([0.40, 0.0, 0.50, 0.0, 1.0, 0.0, 0.0,
    #                                 1.0]).to(env.device)

    # init_jpos = curobo_ik.plan_motion(init_ee_pose[:3],
    #                                   init_ee_pose[3:7]).to(env.device)
    env.reset()
    # curobo_ik.add_obstacle()
    curobo_ik.init_reachability()

    while simulation_app.is_running():
        actions = torch.rand(env.action_space.shape, device=env.device) * 0
        # env.step(actions)
        observation, reward, terminate, time_out, info = env.step(actions)
        curobo_ik.draw_points()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
