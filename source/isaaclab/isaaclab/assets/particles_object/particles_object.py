# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

from ..asset_base import AssetBase
from .particles_object_data import ParticleObjectData
import isaaclab.utils.string as string_utils
if TYPE_CHECKING:
    from .particles_object_cfg import ParticleObjectCfg
# from isaacsim.core.materials import ParticleMaterialView
from isaaclab.sim.spawners.materials import physics_materials
import numpy as np
from pxr import UsdGeom, Sdf, Gf, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.managers.params_generator import ParamsGenerator
import isaaclab.utils.math as math_utils
from isaacsim.core.prims.xform_prim_view import XFormPrimView
from omni.physics.tensors import acquire_tensor_api, float32, uint8, uint32
from typing import Optional, Tuple, Union
import omni.kit.app


class ParticleObject(AssetBase, XFormPrimView):
    """Class for handling particle objects."""

    cfg: ParticleObjectCfg
    """Configuration instance for the particle object."""

    def __init__(self, cfg: ParticleObjectCfg):
        """Initialize the particle object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        XFormPrimView.__init__(self,
                               prim_paths_expr=self.cfg.prim_path,
                               reset_xform_properties=False,
                               name=self.cfg.prim_path.split("/")[-1])

    """
    Properties
    """

    @property
    def data(self) -> ParticleObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset."""
        return 1

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Particle body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    @property
    def max_springs_per_cloth(self) -> int:
        """
        Returns:
            int: maximum number of springs per cloth.
        """
        return self.root_physx_view.max_springs_per_cloth

    @property
    def max_particles_per_cloth(self) -> int:
        """
        Returns:
            int: maximum number of particles per cloth.
        """
        return self.root_physx_view.max_particles_per_cloth

    """
    Operations.
    """

    def random_physical_properties(self, env_ids: Sequence[int] | None = None):
        random_param, normalized_param = self.parames_generator.step_randomize(
            len(env_ids))

        for i, env_idx in enumerate(env_ids):

            physics_materials.spawn_particle_body_material(
                self.cfg.prim_path.replace(".*", str(env_idx.cpu().numpy())) +
                "/PhysicsMaterial",
                sim_utils.ParticleBodyMaterialCfg(
                    **random_param[i], **self.cfg.deform_cfg['fix_params']))

        self._data.physical_params[env_ids] = torch.as_tensor(
            normalized_param, dtype=torch.float32).to(self.device)

    def reset(self, env_ids: Sequence[int] | None = None):

        if env_ids is not None and self.cfg.deform_cfg['random_parmas']:
            self.random_physical_properties(env_ids.cpu())

        # self.root_physx_view.set_spring_damping(
        #     torch.as_tensor(np.random.rand(1, self.max_springs_per_cloth),
        #                     dtype=torch.float32).to(self.device) * 10, env_ids)

    def write_data_to_sim(self):

        import os
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ParticleBodyPropertiesCfg
        from isaaclab.sim import schemas

        current_path = os.getcwd()

    def write_root_pose_to_sim(self,
                               root_pos: torch.Tensor,
                               env_origins: torch.Tensor,
                               env_ids: Sequence[int] | None = None):
        """Set the root pos over selected environment indices into the simulation.

        The root pos comprises of the nodal positions of the simulation mesh for the deformable body.

        Args:
            root_pos: Root poses in simulation frame. Shape is ``(len(env_ids), max_simulation_mesh_vertices_per_body, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """

        tranfomed_points = math_utils.transform_points(
            self._data.default_particle_state_w.clone() -
            env_origins[:, None, :].repeat_interleave(
                self.max_particles_per_cloth, 1), root_pos[:, :3],
            root_pos[:, 3:]) + env_origins[:, None, :].repeat_interleave(
                self.max_particles_per_cloth, 1)
        print(torch.mean(self._data.default_particle_state_w.clone(), dim=1))

        self.root_physx_view.set_positions(tranfomed_points, indices=env_ids)

    def update(self, dt):
        self._data.update(dt)

        self._data.particle_state_w = self.root_physx_view.get_positions(
        ).reshape(self.num_instances, -1, 3)

        self._data.particle_state_v = self.root_physx_view.get_velocities(
        ).reshape(self.num_instances, -1, 3)

        self._data.masses = self.root_physx_view.get_masses()

        self._data.spring_damping = self.root_physx_view.get_spring_damping()

        self._data.spring_stiffness = self.root_physx_view.get_spring_stiffness(
        )

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self,
                                root_state: torch.Tensor,
                                env_ids: Sequence[int] | None = None):
        pass

    def write_root_velocity_to_sim(self,
                                   root_velocity: torch.Tensor,
                                   env_ids: Sequence[int] | None = None):
        pass

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""

        # prim_paths = self.root_physx_view.prim_paths[:self.num_bodies]
        return [self.cfg.prim_path.split("/")[-1]]

    def find_bodies(
            self,
            name_keys: str | Sequence[str],
            preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names,
                                                   preserve_order)

    """
    Internal helper.
    """

    def _initialize_impl(self):

        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(
                f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find first mesh path

        mesh_path = prim_utils.get_prim_path(
            prim_utils.get_first_matching_child_prim(
                template_prim_path,
                lambda p: prim_utils.get_prim_type_name(p) == "Mesh"))
        # resolve mesh path back into regex expression
        mesh_path_expr = self.cfg.prim_path + mesh_path[len(template_prim_path
                                                            ):]
        self.mesh_path_expr = mesh_path_expr

        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        self._root_physx_view = self._physics_sim_view.create_particle_cloth_view(
            mesh_path_expr.replace(".*", "*"))

        self._data = ParticleObjectData(self.root_physx_view, self.device)
        self.parames_generator = ParamsGenerator(
            self.cfg.deform_cfg['params'], self.cfg.deform_cfg['params_range'])

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants

        self._data.particle_state_w = torch.zeros(
            (self.num_instances, self.max_particles_per_cloth, 3))

        self._data.particle_state_v = torch.zeros(
            (self.num_instances, self.max_particles_per_cloth, 3))

        self._data.masses = torch.zeros(
            (self.num_instances, self.max_particles_per_cloth))

        self._data.spring_damping = torch.zeros(
            (self.num_instances, self.max_springs_per_cloth))

        self._data.spring_stiffness = torch.zeros(
            (self.num_instances, self.max_springs_per_cloth))
        self._cloth_auto_apis = [None] * self.num_instances
        self._cloth_apis = [None] * self._count

        self._data.physical_params = torch.zeros(
            self.num_instances,
            len(self.parames_generator.parames_name),
            dtype=torch.float,
            device=self.device)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (tuple(self.cfg.init_state.pos) +
                              tuple(self.cfg.init_state.rot) +
                              tuple(self.cfg.init_state.lin_vel) +
                              tuple(self.cfg.init_state.ang_vel))
        default_root_state = torch.tensor(default_root_state,
                                          dtype=torch.float,
                                          device=self.device)
        self._data.default_root_state = default_root_state.repeat(
            self.num_instances, 1)
        self._data.default_particle_state_w = self.root_physx_view.get_positions(
        ).reshape(self.num_instances, -1, 3).clone()

        self._data.default_particle_state_v = self.root_physx_view.get_velocities(
        ).reshape(self.num_instances, -1, 3).clone()

    """
    Internal simulation callbacks.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(
                    self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    #=======================================================================================
    #### need to debug

    def _apply_cloth_auto_api(self, index):
        if self._cloth_auto_apis[index] is None:
            if self._prims[index].HasAPI(
                    PhysxSchema.PhysxAutoParticleClothAPI):
                cloth_api = PhysxSchema.PhysxAutoParticleClothAPI(
                    self._prims[index])
            else:
                cloth_api = PhysxSchema.PhysxAutoParticleClothAPI.Apply(
                    self._prims[index])
            self._cloth_auto_apis[index] = cloth_api

    def _apply_cloth_api(self, index):
        if self._cloth_apis[index] is None:
            if self._prims[index].HasAPI(PhysxSchema.PhysxParticleClothAPI):
                cloth_api = PhysxSchema.PhysxParticleClothAPI(
                    self._prims[index])
            else:
                cloth_api = PhysxSchema.PhysxParticleClothAPI.Apply(
                    self._prims[index])
            self._cloth_apis[index] = cloth_api

    def _apply_particle_api(self, index):
        if self._cloth_apis[index] is None:
            if self._prims[index].HasAPI(PhysxSchema.PhysxParticleAPI):
                particle_api = PhysxSchema.PhysxParticleAPI(self._prims[index])
            else:
                particle_api = PhysxSchema.PhysxParticleAPI.Apply(
                    self._prims[index])
            self._particle_apis[index] = particle_api

    def set_world_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the particle world positions for the cloths indicated by the indices.

        Args:
            positions (Union[np.ndarray, torch.Tensor]): particle positions with the shape
                                                                                (M, max_particles_per_cloth, 3).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            new_positions = self._backend_utils.move_data(
                positions, self._device)
            current_positions = self.get_world_positions(clone=False)
            current_positions[indices] = new_positions
            self.root_physx_view.set_positions(current_positions, indices)
        else:
            idx_count = 0
            for i in indices:
                self._apply_cloth_auto_api(i.tolist())
                points = self._prims[i.tolist()].GetAttribute("points").Get()
                if points is None:
                    raise Exception(
                        f"The prim {self.name} does not have points attribute."
                    )
                self._prims[i.tolist()].GetAttribute("points").Set(
                    positions[idx_count].tolist())
                idx_count += 1
        return

    def get_world_positions(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the particle world positions for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: position tensor with shape (M, max_particles_per_cloth, 3)
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            positions = self.root_physx_view.get_positions()
            if not clone:
                return positions[indices].reshape(len(indices), -1, 3)
            else:
                return self._backend_utils.clone_tensor(
                    positions[indices].reshape(len(indices), -1, 3),
                    device=self._device)
        else:
            positions = self._backend_utils.create_zeros_tensor(
                [indices.shape[0], self.max_particles_per_cloth, 3],
                dtype="float32",
                device=self._device)
            write_idx = 0
            for i in indices:
                self._apply_cloth_auto_api(i.tolist())
                points = self._prims[i.tolist()].GetAttribute("points").Get()
                if points is None:
                    raise Exception(
                        f"The prim {self.name} does not have points attribute."
                    )
                positions[
                    write_idx] = self._backend_utils.create_tensor_from_list(
                        points, dtype="float32",
                        device=self._device).view(self.max_particles_per_cloth,
                                                  3)
                write_idx += 1
            return positions

    def set_velocities(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the particle velocities for the cloths indicated by the indices.

        Args:
            velocities (Union[np.ndarray, torch.Tensor]): particle velocities with the shape
                                                                                (M, max_particles_per_cloth, 3).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            new_velocities = self._backend_utils.move_data(
                velocities, self._device)
            current_velocities = self.get_velocities(clone=False)
            current_velocities[indices] = new_velocities
            self.root_physx_view.set_velocities(current_velocities, indices)
        else:
            idx_count = 0
            for i in indices:
                self._apply_cloth_auto_api(i.tolist())
                point_velocities = self._prims[i.tolist()].GetAttribute(
                    "velocities").Get()
                if point_velocities is None:
                    raise Exception(
                        f"The prim {self.name} does not have velocities attribute."
                    )
                self._prims[i.tolist()].GetAttribute("velocities").Set(
                    velocities[idx_count].tolist())
                idx_count += 1

    def get_velocities(self,
                       indices: Optional[Union[np.ndarray, list,
                                               torch.Tensor]] = None,
                       clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the particle velocities for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: velocity tensor with shape (M, max_particles_per_cloth, 3)
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            velocities = self.root_physx_view.get_velocities()
            if not clone:
                return velocities[indices].reshape(len(indices), -1, 3)
            else:
                return self._backend_utils.clone_tensor(
                    velocities[indices].reshape(len(indices), -1, 3),
                    device=self._device)
        else:
            velocities = self._backend_utils.create_zeros_tensor(
                [indices.shape[0], self.max_particles_per_cloth, 3],
                dtype="float32",
                device=self._device)
            write_idx = 0
            for i in indices:
                self._apply_cloth_auto_api(i.tolist())
                point_velocities = self._prims[i.tolist()].GetAttribute(
                    "velocities").Get()
                if point_velocities is None:
                    raise Exception(
                        f"The prim {self.name} does not have velocities attribute."
                    )
                velocities[
                    write_idx] = self._backend_utils.create_tensor_from_list(
                        point_velocities, dtype="float32",
                        device=self._device).view(self.max_particles_per_cloth,
                                                  3)
                write_idx += 1
            return velocities

    def set_particle_masses(
        self,
        masses: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the particle masses for the cloths indicated by the indices.

        Args:
            masses (Union[np.ndarray, torch.Tensor]): cloth particle masses with the shape
                                                                                (M, max_particles_per_cloth, 3).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            new_masses = self._backend_utils.move_data(masses, self._device)
            current_masses = self.get_masses(clone=False)
            current_masses[indices] = new_masses
            self.root_physx_view.set_masses(current_masses, indices)
        else:
            idx_count = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        mass_api = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        mass_api = UsdPhysics.MassAPI.Apply(
                            self._prims[i.tolist()])
                    self._mass_apis[i.tolist()] = mass_api
                mass_api.GetMassAttr().Set(sum(masses[idx_count].tolist()))
                idx_count += 1

    def get_particle_masses(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the particle masses for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: mass tensor with shape (M, max_particles_per_cloth)
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            masses = self.root_physx_view.get_masses()
            if not clone:
                return masses[indices]
            else:
                return self._backend_utils.clone_tensor(masses[indices],
                                                        device=self._device)
        else:
            values = self._backend_utils.create_zeros_tensor(
                [indices.shape[0], self.max_particles_per_cloth],
                dtype="float32",
                device=self._device)
            write_idx = 0
            for i in indices:
                if "physics:mass" not in self._prims[
                        i.tolist()].GetPropertyNames():
                    carb.log_warn(
                        f"physics:mass is not defined on the cloth prim: {self.name}. Using the default value."
                    )
                    values[write_idx] = (
                        self._mass_apis[i.tolist()].CreateMassAttr().Get() /
                        self.max_particles_per_cloth)
                else:
                    values[write_idx, :] = (
                        self._mass_apis[i.tolist()].GetMassAttr().Get() /
                        self.max_particles_per_cloth)
                write_idx += 1
            return values

    def set_stretch_stiffnesses(
        self,
        stiffness: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the spring stretch stiffness values for springs within the cloths indicated by the indices.

        Args:
            stiffness (Union[np.ndarray, torch.Tensor]): cloth spring stiffness with the shape  (M, max_springs_per_cloth).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            new_stiffnesses = self._backend_utils.move_data(
                stiffness, self._device)
            current_stiffnesses = self.get_stretch_stiffnesses(clone=False)
            current_stiffnesses[indices] = new_stiffnesses
            self.root_physx_view.set_spring_stiffness(current_stiffnesses,
                                                      indices)
        else:
            idx_count = 0
            for i in indices:
                if stiffness[idx_count].any() < 0:
                    carb.log_error(
                        "The range of stiffness is [0. inf). Incorrect value for index ",
                        idx_count)
                self._apply_cloth_api(i.tolist())
                if "physxParticle:springStiffnesses" not in self._prims[
                        i.tolist()].GetPropertyNames():
                    self._cloth_apis[
                        i.tolist()].CreateSpringStiffnessesAttr().Set(
                            Vt.FloatArray(stiffness[idx_count].tolist()))
                else:
                    self._cloth_apis[
                        i.tolist()].GetSpringStiffnessesAttr().Set(
                            Vt.FloatArray(stiffness[idx_count].tolist()))
                idx_count += 1

    def get_stretch_stiffnesses(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the spring stretch stiffness for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: stiffness tensor with shape (M, max_springs_per_cloth)
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            stiffness = self.root_physx_view.get_spring_stiffness()
            if not clone:
                return stiffness[indices]
            else:
                return self._backend_utils.clone_tensor(stiffness[indices],
                                                        device=self._device)
        else:
            stiffnesses = self._backend_utils.create_zeros_tensor(
                [indices.shape[0], self.max_springs_per_cloth],
                dtype="float32",
                device=self._device)
            write_idx = 0
            for i in indices:
                self._apply_cloth_api(i.tolist())
                if "physxParticle:springStiffnesses" not in self._prims[
                        i.tolist()].GetPropertyNames():
                    carb.log_warn(
                        f"Stretch stiffness is not defined on the cloth prim: {self.name}. Using the default value."
                    )
                    stiffnesses[
                        write_idx] = self._backend_utils.create_tensor_from_list(
                            self._cloth_apis[i.tolist(
                            )].CreateSpringStiffnessesAttr().Get(),
                            dtype="float32")
                else:
                    stiffnesses[
                        write_idx] = self._backend_utils.create_tensor_from_list(
                            self._cloth_apis[
                                i.tolist()].GetSpringStiffnessesAttr().Get(),
                            dtype="float32")
                write_idx += 1
            return stiffnesses

    def set_spring_dampings(
        self,
        damping: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the spring damping for the cloths indicated by the indices.

        Args:
            damping (Union[np.ndarray, torch.Tensor]): cloth spring damping with the shape
                                                                            (M, max_springs_per_cloth).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            new_damping = self._backend_utils.move_data(damping, self._device)
            current_damping = self.get_spring_dampings(clone=False)
            current_damping[indices] = new_damping
            self.root_physx_view.set_spring_damping(current_damping, indices)
        else:
            idx_count = 0
            for i in indices:
                if damping[idx_count].any() < 0:
                    carb.log_error(
                        "The range of damping is [0. inf). Incorrect value for index ",
                        idx_count)
                self._apply_cloth_api(i.tolist())
                if "physxParticle:springDampings" not in self._prims[
                        i.tolist()].GetPropertyNames():
                    self._cloth_apis[
                        i.tolist()].CreateSpringDampingsAttr().Set(
                            Vt.FloatArray(damping[idx_count].tolist()))
                else:
                    self._cloth_apis[i.tolist()].GetSpringDampingsAttr().Set(
                        Vt.FloatArray(damping[idx_count].tolist()))
                idx_count += 1

    def get_spring_dampings(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the spring damping for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: damping tensor with shape (M, max_springs_per_cloth)
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        if not omni.timeline.get_timeline_interface().is_stopped(
        ) and self.root_physx_view is not None:
            damping = self.root_physx_view.get_spring_damping()
            if not clone:
                return damping[indices]
            else:
                return self._backend_utils.clone_tensor(damping[indices],
                                                        device=self._device)
        else:
            dampings = self._backend_utils.create_zeros_tensor(
                [indices.shape[0], self.max_springs_per_cloth],
                dtype="float32",
                device=self._device)
            write_idx = 0
            for i in indices:
                self._apply_cloth_api(i.tolist())
                if "physxParticle:springDampings" not in self._prims[
                        i.tolist()].GetPropertyNames():
                    carb.log_warn(
                        f"Stretch damping is not defined on the cloth prim: {self.name}. Using the default value"
                    )
                    dampings[
                        write_idx] = self._backend_utils.create_tensor_from_list(
                            self._cloth_apis[
                                i.tolist()].GetSpringDampingsAttr().Get(),
                            dtype="float32")
                else:
                    dampings[
                        write_idx] = self._backend_utils.create_tensor_from_list(
                            self._cloth_apis[
                                i.tolist()].GetSpringDampingsAttr().Get(),
                            dtype="float32")
                write_idx += 1
            return dampings

    def set_pressures(
        self,
        pressures: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the pressures of the cloths indicated by the indices.

        Args:
            pressures (Union[np.ndarray, torch.Tensor]): cloths pressure with shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            self._apply_cloth_api(i.tolist())
            if "physxParticle:pressure" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_apis[i.tolist()].CreatePressureAttr().Set(
                    pressures[idx_count].tolist())
            else:
                self._cloth_apis[i.tolist()].GetPressureAttr().Set(
                    pressures[idx_count].tolist())
            idx_count += 1

    def set_self_collision_filters(
        self,
        self_collision_filters: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the self collision filters for the cloths indicated by the indices.

        Args:
            self_collision_filters (Union[np.ndarray, torch.Tensor]): self collision filters with the shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            self._apply_cloth_api(i.tolist())
            if "physxParticle:selfCollisionFilter" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_apis[
                    i.tolist()].CreateSelfCollisionFilterAttr().Set(
                        self_collision_filters[idx_count].tolist())
            else:
                self._cloth_apis[i.tolist()].GetSelfCollisionFilterAttr().Set(
                    self_collision_filters[idx_count].tolist())
            idx_count += 1

    def set_self_collisions(
        self,
        self_collisions: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the self collision flags for the cloths indicated by the indices.

        Args:
            self_collisions (Union[np.ndarray, torch.Tensor]): self collision flag with the shape  (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            self._apply_particle_api(i.tolist())
            if "physxParticle:selfCollision" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._particle_apis[i.tolist()].CreateSelfCollisionAttr().Set(
                    self_collisions[idx_count].tolist())
            else:
                self._particle_apis[i.tolist()].GetSelfCollisionAttr().Set(
                    self_collisions[idx_count].tolist())
            idx_count += 1

    def set_particle_groups(
        self,
        particle_groups: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets the particle group of the cloths indicated by the indices.

        Args:
            particle_groups (Union[np.ndarray, torch.Tensor]): particle group with shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            self._apply_particle_api(i.tolist())
            if "physxParticle:particleGroup" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._particle_apis[i.tolist()].CreateParticleGroupAttr().Set(
                    particle_groups[idx_count].tolist())
            else:
                self._particle_apis[i.tolist()].GetParticleGroupAttr().Set(
                    particle_groups[idx_count].tolist())
            idx_count += 1

    def set_cloths_dampings(
        self,
        values: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets a single value of damping to all the springs within cloths indicated by the indices.

        Args:
            values (Union[np.ndarray, torch.Tensor]): cloth spring damping with the shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            if values[idx_count] < 0:
                carb.log_error(
                    "The range of damping is [0. inf). Incorrect value for index ",
                    idx_count)
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springDamping" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_auto_apis[
                    i.tolist()].CreateSpringDampingAttr().Set(
                        values[idx_count].tolist())
            else:
                self._cloth_auto_apis[i.tolist()].GetSpringDampingAttr().Set(
                    values[idx_count].tolist())
            idx_count += 1

    def set_cloths_stretch_stiffnesses(
        self,
        values: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets a single value of stretch stiffnesses to all the springs within cloths indicated by the indices.

        Args:
            values (Union[np.ndarray, torch.Tensor]): cloth spring stretch stiffness values with the shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            if values[idx_count] < 0:
                carb.log_error(
                    "The range of stretch stiffness is [0. inf). Incorrect value for index ",
                    idx_count)
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springStretchStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_auto_apis[
                    i.tolist()].CreateSpringStretchStiffnessAttr().Set(
                        values[idx_count].tolist())
            else:
                self._cloth_auto_apis[
                    i.tolist()].GetSpringStretchStiffnessAttr().Set(
                        values[idx_count].tolist())

            idx_count += 1

    def set_cloths_bend_stiffnesses(
        self,
        values: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets a single value of bend stiffnesses to all the springs within cloths indicated by the indices.

        Args:
            values (Union[np.ndarray, torch.Tensor]): cloth spring bend stiffness values with the shape (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            if values[idx_count] < 0:
                carb.log_error(
                    "The range of bend stiffness is [0. inf). Incorrect value for index ",
                    idx_count)
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springBendStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_auto_apis[
                    i.tolist()].CreateSpringBendStiffnessAttr().Set(
                        values[idx_count])
            else:
                self._cloth_auto_apis[
                    i.tolist()].GetSpringBendStiffnessAttr().Set(
                        values[idx_count])

            idx_count += 1

    def set_cloths_shear_stiffnesses(
        self,
        values: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """Sets a single value of shear stiffnesses to all the springs within cloths indicated by the indices.

        Args:
            values (Union[np.ndarray, torch.Tensor]): cloth spring shear stiffness values with the shape  (M, ).
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices,
                                                      self.count,
                                                      device=self._device)
        idx_count = 0
        for i in indices:
            if values[idx_count] < 0:
                carb.log_error(
                    "The range of shear stiffness is [0. inf). Incorrect value for index ",
                    idx_count)
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springShearStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                self._cloth_auto_apis[
                    i.tolist()].CreateSpringShearStiffnessAttr().Set(
                        values[idx_count])
            else:

                self._cloth_auto_apis[
                    i.tolist()].GetSpringShearStiffnessAttr().Set(
                        values[idx_count])

            idx_count += 1

    def get_cloths_dampings(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the value of damping set for all the springs within cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: damping tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        values = self._backend_utils.create_zeros_tensor([indices.shape[0]],
                                                         dtype="float32",
                                                         device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springDamping" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"damping is not defined on the cloth prim: {self.name}. Using the default value."
                )
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].CreateSpringDampingAttr().Get()
            else:
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].GetSpringDampingAttr().Get()
            write_idx += 1
        return values

    def get_cloths_stretch_stiffnesses(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the value of stretch stiffness set to all the springs within cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: stretch stiffness tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        values = self._backend_utils.create_zeros_tensor([indices.shape[0]],
                                                         dtype="float32",
                                                         device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springStretchStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"Stretch stiffness is not defined on the cloth prim: {self.name}. Using the default value."
                )
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].CreateSpringStretchStiffnessAttr().Get()
            else:
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].GetSpringStretchStiffnessAttr().Get()
            write_idx += 1
        return values

    def get_cloths_bend_stiffnesses(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the value of bend stiffness set to all the springs within cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: bend stiffness tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        values = self._backend_utils.create_zeros_tensor([indices.shape[0]],
                                                         dtype="float32",
                                                         device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springBendStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"bend stiffness is not defined on the cloth prim: {self.name}. Using the default value."
                )
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].CreateSpringBendStiffnessAttr().Get()
            else:
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].GetSpringBendStiffnessAttr().Get()
            write_idx += 1
        return values

    def get_cloths_shear_stiffnesses(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the value of shear stiffness set to all the springs within cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: shear stiffness tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        values = self._backend_utils.create_zeros_tensor([indices.shape[0]],
                                                         dtype="float32",
                                                         device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_auto_api(i.tolist())
            if "physxAutoParticleCloth:springShearStiffness" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"shear stiffness is not defined on the cloth prim: {self.name}. Using the default values."
                )
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].CreateSpringShearStiffnessAttr().Get()
            else:
                values[write_idx] = self._cloth_auto_apis[
                    i.tolist()].GetSpringShearStiffnessAttr().Get()
            write_idx += 1
        return values

    def get_self_collision_filters(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the self collision filters for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: the self collision filters tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        self_collision_filters = self._backend_utils.create_zeros_tensor(
            [indices.shape[0]], dtype="bool", device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_api(i.tolist())
            if "physxParticle:selfCollisionFilter" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"selfCollisionFilter is not defined on the cloth prim: {self.name}. Using the default values."
                )
                self_collision_filters[write_idx] = self._cloth_apis[
                    i.tolist()].CreateSelfCollisionFilterAttr().Get()
            else:
                self_collision_filters[write_idx] = self._cloth_apis[
                    i.tolist()].GetSelfCollisionFilterAttr().Get()
            write_idx += 1
        return self_collision_filters

    def get_self_collisions(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the self collision for the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: the self collision tensor with shape (M, )
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        self_collisions = self._backend_utils.create_zeros_tensor(
            [indices.shape[0]], dtype="bool", device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_particle_api(i.tolist())
            if "physxParticle:selfCollision" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"selfCollision is not defined on the cloth prim: {self.name}. Using the default values."
                )
                self_collisions[write_idx] = self._particle_apis[
                    i.tolist()].CreateSelfCollisionAttr().Get()
            else:
                self_collisions[write_idx] = self._particle_apis[
                    i.tolist()].GetSelfCollisionAttr().Get()
            write_idx += 1
        return self_collisions

    def get_pressures(self,
                      indices: Optional[Union[np.ndarray, list,
                                              torch.Tensor]] = None,
                      clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the pressures of the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: cloths pressure with shape (M, ).
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        pressures = self._backend_utils.create_zeros_tensor(
            [indices.shape[0]], dtype="float32", device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_cloth_api(i.tolist())
            if "physxParticle:pressure" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"pressure is not defined on the cloth prim: {self.name}. Using the default value."
                )
                pressures[write_idx] = self._cloth_apis[
                    i.tolist()].CreatePressureAttr().Get()
            else:
                pressures[write_idx] = self._cloth_apis[
                    i.tolist()].GetPressureAttr().Get()
            write_idx += 1
        return pressures

    def get_particle_groups(
            self,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
            clone: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Gets the particle groups of the cloths indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which cloth prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: particle groups with shape (M, ).
        """
        indices = self._backend_utils.resolve_indices(indices, self.count,
                                                      self._device)
        particle_groups = self._backend_utils.create_zeros_tensor(
            [indices.shape[0]], dtype="int32", device=self._device)
        write_idx = 0
        for i in indices:
            self._apply_particle_api(i.tolist())
            if "physxParticle:particleGroup" not in self._prims[
                    i.tolist()].GetPropertyNames():
                carb.log_warn(
                    f"particleGroup is not defined on the cloth prim: {self.name}. Using the default value."
                )
                particle_groups[write_idx] = self._particle_apis[
                    i.tolist()].GetParticleGroupAttr().Get()
            else:
                particle_groups[write_idx] = self._particle_apis[
                    i.tolist()].GetParticleGroupAttr().Get()
            write_idx += 1
        return particle_groups
