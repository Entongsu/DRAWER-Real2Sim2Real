# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import sys

sys.path.append("submodule/stable-baselines3")
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""
"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

parser.add_argument(
    "--seg_handle",
    action="store_true",
    default=False,
)

# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.automatic_articulation.utils.data_wrapper import Datawrapper


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
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
    save_config["params"]["Camera"]["initial"] = True
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    init_grasp = args_cli.init_grasp
    init_open = args_cli.init_open
    init_placement = args_cli.init_placement
    init_close = args_cli.init_close
    replay_normalized_actions = args_cli.replay
    collect_data = True if not replay_normalized_actions else False

    multi_env = Datawrapper(
        env,
        collision_checker=True,
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
        seg_handle=args_cli.seg_handle,
        use_fps=True)
    env.reset()

    stop = False

    for i in range(args_cli.num_demos):  #multi_env.num_grasp_demos

        if not replay_normalized_actions:
            stop = multi_env.step_unnormalized_env(skip_frame=2)
        else:
            stop = multi_env.replay_normalized_action()

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
