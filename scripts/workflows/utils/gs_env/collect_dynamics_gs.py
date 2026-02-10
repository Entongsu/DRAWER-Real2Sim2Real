# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to collect demonstrations with Isaac Lab environments."""
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect demonstrations for Isaac Lab environments.")

parser.add_argument("--num_envs",
                    type=int,
                    default=1,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument("--num_demos",
                    type=int,
                    default=1,
                    help="Number of episodes to store in the dataset.")
parser.add_argument("--filename",
                    type=str,
                    default="hdf_dataset",
                    help="Basename of output file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.io import dump_pickle, dump_yaml

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.Plush import mdp
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tools.deformable_obs import *


def pre_process_actions(delta_pose: torch.Tensor,
                        gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1),
                                  dtype=torch.float,
                                  device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # assert (
    #     args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    # ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs)
    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.time_out = DoneTerm(func=mdp.time_out)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # specify directory for logging experiments
    log_dir = f"./logs/{args_cli.filename}/dynamics_gs/"
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
        env_config={"device": args_cli.device},
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces

    collector_interface.reset()
    index = 0
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():

            actions = torch.rand(env.action_space.shape,
                                 device=env.unwrapped.device) * 0
            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping

            # -- obs
            index += 1
            if index % 1 == 0:

                randomize_camera_pose(
                    env, torch.as_tensor(torch.arange(0, env.num_envs)),
                    env.episode_length_buf * 0.4)
                collector_interface.add("actions", actions)
                for key, value in obs_dict["policy"].items():

                    collector_interface.add(f"obs/{key}", value)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

            # check if enough data is collected
            if collector_interface.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
