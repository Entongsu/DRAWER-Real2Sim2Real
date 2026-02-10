# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.droid_uw import mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.droid_uw.droid_uw_cfg import DroidnUWCfg
##
# Pre-defined configs
##

from dataclasses import dataclass, field
from isaaclab.utils import configclass

import sys
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm

sys.path.append(".")
from scripts.workflows.utils.config_setting import set_from_config
import os
from scripts.workflows.utils.robot_cfg import DROID_CFG
import torch

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.droid_uw.mdp.obs_reward_buffer import RewardObsBuffer
from isaaclab.sensors import ContactSensorCfg


@configclass
class UWDroidnEnvCfg(DroidnUWCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):

        env_cfg = self.config_yaml
        self.env_config = env_cfg

        # post init of parent
        super().__post_init__()
        # self.config_yaml = env_cfg
        current_path = os.getcwd()
        if self.env_config["params"]["Task"]["robot_type"] == "franka":
            self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot")

            self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": -0.0},
            )
            left_finger_name = "panda_leftfinger"
            right_finger_name = "panda_rightfinger"
            ee_link_name = "panda_hand"
        else:
            self.scene.robot = DROID_CFG

            self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
                asset_name="robot",
                joint_names=[
                    'finger_joint',
                ],
            )
            left_finger_name = "left_inner_finger"
            right_finger_name = "right_inner_finger"
            ee_link_name = "panda_link7"

        # Set Actions for the specific robot type (franka)

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            preserve_order=True,
            # use_default_offset=False,
        )

        camera_setting = env_cfg["params"]["Camera"]

        if camera_setting["initial"]:
            setattr(self.observations.policy, "camera_obs", "camera_obs")
            self.observations.policy.camera_obs = ObsTerm(
                func=mdp.process_camera_data,
                params={
                    "whole_rgb": camera_setting["whole_rgb"],
                    "seg_rgb": camera_setting["seg_rgb"],
                    "whole_pc": camera_setting["whole_pc"],
                    "seg_pc": camera_setting["seg_pc"],
                    "bbox": camera_setting.get("bbox"),
                    "segmentation_name": camera_setting["segmentation_name"],
                    "max_length": camera_setting["max_length"],
                    "align_robot_base": camera_setting["align_robot_base"],
                    "whole_depth": camera_setting["whole_depth"],
                },
            )

        set_from_config(self.scene, env_cfg, current_path)
        if env_cfg["params"]["GS_Camera"]["initial"]:
            from scripts.workflows.utils.gs_env.multi_gs_env import MultiGaussianEnv
            multi_gs_group = MultiGaussianEnv(env_config=env_cfg)
            setattr(self.observations.policy, "gs_image", "gs_image")

            self.observations.policy.gs_image = ObsTerm(
                func=multi_gs_group.extract_gaussians)

        if env_cfg["params"]["RL_Train"]["initial"]:
            for rigid_name in env_cfg["params"]["RL_Train"][
                    "rigid_object_names"]:
                state_info = ObsTerm(mdp.get_root_state,
                                     params={"name": rigid_name})
                setattr(self.observations.policy, f"{rigid_name}_state",
                        state_info)
        else:
            for rigid_name in env_cfg["params"]["Task"]["reset_object_names"]:
                state_info = ObsTerm(mdp.get_root_state,
                                     params={"name": rigid_name})
                setattr(self.observations.policy, f"{rigid_name}_state",
                        state_info)

        if env_cfg["params"]["GS_Camera"]["initial"]:

            for rigid_name in env_cfg["params"]["RigidObject"].keys():
                state_info = ObsTerm(mdp.get_root_state,
                                     params={"name": rigid_name})
                setattr(self.observations.policy, f"{rigid_name}_state",
                        state_info)

        setattr(self.observations.policy, "ee_pose",
                ObsTerm(func=mdp.ee_pose, params={"body_name": ee_link_name}))
        target_object_name = env_cfg["params"]["Task"]["target_object"]

        RewardObsBuffer(self.rewards,
                        self.observations.policy,
                        self.config_yaml,
                        target_object_name=target_object_name,
                        placement_object_name=env_cfg["params"]["Task"]
                        ["placement"]["placement_object"],
                        ee_frame_name=ee_link_name,
                        left_finger_name=left_finger_name,
                        right_finger_name=right_finger_name,
                        use_gripper_offset=True)

        self.scene.contact_right_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + target_object_name])

        self.scene.contact_left_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + target_object_name],
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
