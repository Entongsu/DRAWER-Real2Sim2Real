# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .particles_object import ParticleObject
from isaacsim.core.materials import ParticleMaterialView


@configclass
class ParticleObjectCfg(AssetBaseCfg):
    """Configuration parameters for a Particle object."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    class_type: type = ParticleObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""
