# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
import omni.physics.tensors.impl.api as physx
import weakref

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


@dataclass
class ParticleObjectData:
    """Data container for a robot."""

    default_particle_state_v: torch.Tensor = None

    def __init__(self, root_physx_view: physx.ParticleClothView, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view of the object.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        self._root_physx_view = weakref.proxy(
            root_physx_view)  # weak reference to avoid circular references
        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Obtain global physics sim view
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]),
                                   device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor(
            (1.0, 0.0, 0.0),
            device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize buffers for finite differencing
        self._previous_body_vel_w = torch.zeros(
            (self._root_physx_view.count, 1, 6), device=self.device)

        # Initialize the lazy buffers.
        self._root_state_w = TimestampedBuffer()
        self._body_acc_w = TimestampedBuffer()

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13)."""
        # if self._root_state_w.timestamp < self._sim_timestamp:
        #     # read data from simulation

        #     pose = self._root_physx_view.get_transforms().clone()
        #     pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
        #     velocity = self._root_physx_view.get_velocities()
        #     # set the buffer data and timestamp
        #     self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
        #     self._root_state_w.timestamp = self._sim_timestamp
        # return self._root_state_w.data
        pass

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        """Nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, max_simulation_mesh_vertices_per_body, 3)``.
        """
        return self.nodal_state_w[:, :self.nodal_state_w.size(1) // 2, :]

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        """Vertex velocities for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, max_simulation_mesh_vertices_per_body, 3)``.
        """
        return self.nodal_state_w[:, self.nodal_state_w.size(1) // 2:, :]

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, 3)``.
        """
        return self.nodal_pos_w.mean(dim=1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, 3)``.
        """
        return self.nodal_vel_w.mean(dim=1)

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        self._sim_timestamp += dt
        # Trigger an update of the body acceleration buffer at a higher frequency
        # since we do finite differencing.
        self._body_acc_w
