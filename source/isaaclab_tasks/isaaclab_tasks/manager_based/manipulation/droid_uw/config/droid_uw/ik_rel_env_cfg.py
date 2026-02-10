# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
from scripts.workflows.utils.robot_cfg import DROID_CFG
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


@configclass
class UWDroidnEnvCfg(joint_pos_env_cfg.UWDroidnEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        if self.env_config["params"]["Task"]["robot_type"] == "franka":
            # self.scene.robot = FRANKA_PANDA_CFG.replace(
            #     prim_path="{ENV_REGEX_NS}/Robot")
            ee_link = "panda_hand"
        else:
            # self.scene.robot = DROID_CFG
            ee_link = "panda_link8"

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name=ee_link,
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=True,
                                                   ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )


@configclass
class UWDroidnEnvCfg_PLAY(UWDroidnEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
