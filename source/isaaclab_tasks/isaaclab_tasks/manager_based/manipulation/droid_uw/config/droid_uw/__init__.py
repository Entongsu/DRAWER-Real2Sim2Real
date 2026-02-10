# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-UW-Droid-Kitchen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{joint_pos_env_cfg.UWDroidnEnvCfg}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:DroidPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-UW-Droid-Kitchen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{joint_pos_env_cfg.UWDroidnEnvCfg}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-UW-Droid-Kitchen-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_abs_env_cfg.UWDroidnEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-UW-Droid-Kitchen-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ik_rel_env_cfg.UWDroidnEnvCfg}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:DroidPPORunnerCfg",
    },
    disable_env_checker=True,
)
