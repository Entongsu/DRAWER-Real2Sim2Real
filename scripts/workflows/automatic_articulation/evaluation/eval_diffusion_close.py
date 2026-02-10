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

from scripts.workflows.automatic_articulation.utils.data_wrapper import Datawrapper
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.automatic_articulation.utils.replay_datawrapper import ReplayDatawrapper


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

    init_grasp = False
    init_open = False
    init_placement = False
    init_close = True
    collect_data = False

    multi_env = ReplayDatawrapper(
        env,
        init_grasp=init_grasp,
        init_open=init_open,
        init_placement=init_placement,
        init_close=init_close,
        collect_data=collect_data,
        args_cli=args_cli,
        env_config=save_config,
        filter_keys=["segmentation", "seg_pc", "rgb", "seg_rgb"],
        policy_path=
        "../robomimic/bc_trained_models/diffusion/20241030152953/models/model_epoch_320.pth"
    )
    env.reset()
    stats = np.load(args_cli.log_dir + "/close_stats.npy", allow_pickle=True)
    total_count = 0
    success_count = 0
    import h5py
    normalized_grasp = h5py.File(f"{args_cli.log_dir}/close_normalized2.hdf5",
                                 'r+')

    while simulation_app.is_running():
        multi_env.extract_data()
        # open cabinet
        #=========================================================================================================
        last_obs = multi_env.reset_env()

        success = multi_env.eval_env(last_obs,
                                     stats,
                                     normalized_grasp,
                                     horiozon=60)
        #multi_env.eval_open_loop_env(last_obs)
        #=========================================================================================================
        env.reset()
        total_count += 1
        if success:
            success_count += 1
        print(
            f"Success rate: {success_count}/{total_count} = {success_count/total_count}"
        )

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
