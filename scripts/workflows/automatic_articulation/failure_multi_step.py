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

from scripts.workflows.automatic_articulation.utils.failure_data_wrapper import FailureDatawrapper


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    """Random actions agent with Isaac Lab environment."""
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    # create environment configuration
    replay_normalized_actions = args_cli.replay
    if replay_normalized_actions:
        save_config["params"]["Camera"]["initial"] = True

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    init_grasp = args_cli.init_grasp
    init_open = args_cli.init_open
    init_placement = args_cli.init_placement
    init_close = args_cli.init_close

    collect_data = True

    multi_env = FailureDatawrapper(
        env,
        collision_checker=False,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        init_grasp=init_grasp,
        init_open=init_open,
        init_placement=init_placement,
        init_close=init_close,
        collect_data=collect_data,
        args_cli=args_cli,
        env_config=save_config,
        filter_keys=["segmentation", "seg_rgb", "seg_pc"],
        save_path=args_cli.save_path,
        load_path=args_cli.load_path,
        use_joint_pos="IK" not in args_cli.task,
        use_demo_data=True,
        use_fps=True,
        faiulure_type=args_cli.failure_type,
        failure_attempt=2)
    env.reset()

    stop = False
    for i in range(args_cli.num_demos):  #multi_env.num_grasp_demos

        if not replay_normalized_actions:
            stop = multi_env.step_failure_env(skip_frame=1)
        else:
            stop = multi_env.step_unnormalized_env(
                skip_frame=2,
                start_frame=multi_env.start_frame,
                succes_bool=False)

        #=========================================================================================================
        env.reset()
        if stop:
            break
    # if init_grasp and multi_env.collector_interface.grasp_collector_interface is not None:
    #     multi_env.collector_interface.grasp_collector_interface.close()
    # if init_open and multi_env.collector_interface.cabinet_collector_interface is not None:
    #     multi_env.collector_interface.cabinet_collector_interface.close()
    # multi_env.collector_interface.split_set()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
