import weakref
from scripts.workflows.automatic_articulation.task.env_grasp import GrasperEnv
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.managers import SceneEntityCfg
import torch
import isaaclab.utils.math as math_utils
from tools.curobo_planner import IKPlanner
from tools.curobo_planner import MotionPlanner
from scripts.workflows.automatic_articulation.task.env_cabinet import CabinetOpenEnv
from scripts.workflows.automatic_articulation.task.env_placement import PlacementEnv
from scripts.workflows.automatic_articulation.task.env_close import EnvCloseCabinet
import numpy as np
from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles
from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose
from scripts.workflows.automatic_articulation.utils.process_action import process_action

from scripts.workflows.automatic_articulation.utils.map_env import env_map, collect_map, reset_map, init_setting


class MultiStepEnv:

    def __init__(self,
                 env,
                 use_relative_pose=False,
                 collision_checker=True,
                 init_grasp=False,
                 init_open=False,
                 init_placement=False,
                 init_close=False,
                 env_config=None,
                 sample_interval=5):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.kitchen = env.scene["kitchen"]
        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker
        self.sample_interval = sample_interval

        self.init_grasp = init_grasp
        self.init_open = init_open
        self.init_placement = init_placement
        self.init_close = init_close
        self.env_config = env_config

        init_setting(self)
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)
        self.init_planner()

        self.curo_ik_planner = IKPlanner(env)
        self.initialize_envs()

    def init_env(self, env_class, attr_name):
        setattr(
            self, attr_name,
            env_class(self.env,
                      self.planner,
                      use_relative_pose=self.use_relative_pose,
                      collision_checker=self.collision_checker,
                      env_config=self.env_config))

    def initialize_envs(self):
        if self.init_open:
            self.init_env(CabinetOpenEnv, 'env_cabinet')
        if self.init_grasp:
            self.init_env(GrasperEnv, 'env_grasp')
        if self.init_placement:
            self.init_env(PlacementEnv, 'env_placement')
        if self.init_close:
            self.init_env(EnvCloseCabinet, 'env_cabinetclose')

        # Clear CUDA cache
        torch.cuda.empty_cache()

    def init_planner(self):
        self.planner = MotionPlanner(
            self.env,
            collision_checker=self.collision_checker,
            reference_prim_path="/World/envs/env_0/Robot",
            ignore_substring=[
                "/World/envs/env_0/Robot",
                "/World/GroundPlane",
                "/World/collisions",
                "/World/light",
                "/curobo",
            ],
        )

    def reset_rigid_articulation(self,
                                 target_name,
                                 pose_range,
                                 reset_grasp=True):

        random_position, random_orientaton = mdp.reset_root_state_uniform(
            self.env,
            env_ids=self.env_ids,
            pose_range=pose_range,
            velocity_range={},
            asset_cfg=SceneEntityCfg(target_name))
        return random_position, random_orientaton

    def reset_kitchen_drawer(self, init_grasp=False):
        default_jpos = self.kitchen._data.default_joint_pos.clone()

        if init_grasp and not self.init_close:
            limit_index = -1 if self.target_joint_type == "revolute" else -1
            default_jpos[:, self.
                         joint_ids] = self.kitchen._data.joint_limits[:, self.
                                                                      joint_ids,
                                                                      limit_index]

        elif self.init_open:
            default_jpos[:, self.joint_ids] = 0.0
        elif self.init_close:

            default_jpos[:, self.
                         joint_ids] = self.kitchen._data.joint_limits[:, self.
                                                                      joint_ids,
                                                                      1]

        self.kitchen._data.reset_joint_pos = default_jpos
        self.kitchen.root_physx_view.set_dof_positions(default_jpos,
                                                       self.env_ids)

    def add_noise_to_position(self, ee_pose):

        # bias these values randomly
        position_range = self.ee_random_range["position_range"]

        ee_pose[:, :3] += math_utils.sample_uniform(-position_range[0],
                                                    position_range[0],
                                                    ee_pose[:, :3].shape,
                                                    ee_pose.device)
        delta_euler_angle_noise = math_utils.sample_uniform(
            -position_range[1], position_range[1], ee_pose[:, :3].shape,
            ee_pose.device)
        delta_quat_noise = math_utils.quat_from_euler_xyz(
            delta_euler_angle_noise[:, 0], delta_euler_angle_noise[:, 1],
            delta_euler_angle_noise[:, 2])
        ee_pose[:, 3:7] = math_utils.quat_mul(delta_quat_noise, ee_pose[:,
                                                                        3:7])

        return ee_pose

    def add_noise_to_jpos(self, joint_pos):
        asset = self.robot

        # bias these values randomly
        position_range = self.joint_random_range["position_range"]
        delta_noise = math_utils.sample_uniform(position_range[0],
                                                position_range[1],
                                                joint_pos.shape,
                                                joint_pos.device)
        joint_pos += delta_noise

        # clamp joint pos to limits
        joint_pos_limits = asset.data.soft_joint_pos_limits[self.env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0],
                                     joint_pos_limits[..., 1])

        return joint_pos

    def random_camera_pose(self):

        for key in self.env.scene.sensors:
            if "camera" in key:
                init_pose = self.env.scene.sensors[key].init_poses
                init_quat = self.env.scene.sensors[key].init_quats
                init_pose = init_pose + (
                    torch.rand(init_pose.shape, device=self.device) -
                    0.0) * self.randomize_camera_pose_range[0]
                init_orientation = torch.cat(
                    math_utils.euler_xyz_from_quat(init_quat)).unsqueeze(0)

                init_orientation += (
                    torch.rand(  # random orientation
                        init_orientation.shape,
                        device=self.device) -
                    0.5) * self.randomize_camera_pose_range[1]

                init_quat = math_utils.quat_from_euler_xyz(
                    init_orientation[:, 0], init_orientation[:, 1],
                    init_orientation[:, 2])

                self.env.scene.sensors[key]._view.set_world_poses(
                    init_pose, init_quat, self.env_ids)

    def reset_joint_pos(
        self,
        init_open=False,
        init_close=False,
        init_grasp=False,
    ):

        # Iterate through the map and apply the corresponding logic
        for flag, env_attr in env_map.items():
            if locals()[flag]:  # Check if the corresponding flag is True
                env = getattr(self, env_attr, None)
                if env is None:
                    raise AttributeError(
                        f"{self} does not have attribute '{env_attr}'")

                # Add noise to the position and set DOF positions
                self.reset_robot_ee_pose = self.add_noise_to_position(
                    env.init_ee_pose.clone())
                self.robot.root_physx_view.set_dof_positions(
                    env.init_jpos.repeat_interleave(self.env.num_envs, 0),
                    self.env_ids)
                break

    def step_reset(self,
                   init_open=False,
                   init_close=False,
                   init_grasp=False,
                   random_joint=True):

        observation, _ = self.env.reset()
        if random_joint:
            self.reset_joint_pos(init_open, init_close, init_grasp)

        if self.randomize_camera_pose:
            self.random_camera_pose()
        for i in range(50):  # reset for stable initial status

            if i < 10:
                # if not init_open and not init_close:  # set  the position of the object when need to grasp

                self.env.scene[self.grasp_object].write_root_pose_to_sim(
                    torch.cat([
                        self.object_random_position,
                        self.object_random_orientation
                    ],
                              dim=1), self.env_ids)
                self.kitchen.root_physx_view.set_dof_positions(
                    self.kitchen._data.reset_joint_pos, self.env_ids)

            if self.use_relative_pose:
                observation, reward, terminate, time_out, info = self.env.step(
                    torch.rand(self.env.action_space.shape, device=self.device)
                    * 0.0)
            else:

                observation, reward, terminate, time_out, info = self.env.step(
                    self.reset_robot_ee_pose.repeat_interleave(
                        self.env.num_envs, 0))

        return observation

    def reset_all_env(self,
                      reset_grasp=False,
                      reset_cabinet=False,
                      reset_close=False,
                      reset_all=True):

        if reset_all:
            self.reset_kitchen_drawer(
                init_grasp=False if reset_cabinet else True)

            self.reset_rigid_articulation("robot",
                                          self.robot_pose_random_range)

            if self.kitchen_pose_range is not None:
                self.reset_rigid_articulation("kitchen",
                                              self.kitchen_pose_range)

            self.object_random_position, self.object_random_orientation = self.reset_rigid_articulation(
                self.grasp_object,
                self.env.scene[self.grasp_object].cfg.rigid_cfg["pose_range"],
                reset_grasp=True)

            observation = self.step_reset(init_grasp=reset_grasp,
                                          init_close=reset_close,
                                          init_open=reset_cabinet)

            for flag, env_attr in reset_map.items():
                if locals()[flag]:  # Check if the corresponding flag is True
                    env = getattr(self, env_attr, None)
                    if env is None:
                        raise AttributeError(
                            f"{self} does not have attribute '{env_attr}'")
                    result = env.reset(observation)
                    if result is not None:
                        return observation
                    else:
                        return None
        else:

            return self.step_reset(init_grasp=reset_grasp,
                                   init_close=reset_close,
                                   init_open=reset_cabinet)

    def step_manipulate(self,
                        target_pose,
                        collect_grasp=False,
                        collect_placement=False,
                        collect_cabinet=False,
                        collect_close=False):

        # Process actions
        actions = process_action(target_pose, self.use_relative_pose,
                                 self.robot, self.device)
        actions[:, -1] = torch.sign(actions[:, -1] + 0.3)

        # Perform an environment step
        observation, reward, terminate, time_out, info = self.env.step(actions)

        # Check which collection flag is True
        for flag, env_attr in collect_map.items():

            if locals()[flag]:  # Check if the corresponding flag is True
                env = getattr(self, env_attr, None)
                if env is None:
                    raise AttributeError(
                        f"{self} does not have attribute '{env_attr}'")

                # Return with success status from the corresponding environment
                return observation, reward, terminate, time_out, info, actions, env.success_or_not(
                    observation)

        # Default return if no flag is set (optional, if you want a fallback)
        return observation, reward, terminate, time_out, info, actions, None

    #=========================================================================================================
    # replay data
    #=========================================================================================================

    def reset_demo_env(self, demo, use_joint_pos=False):
        obs = demo["obs"]

        # extract init robot setting
        init_joint_pos = torch.as_tensor(obs["joint_pos"][0]).to(
            self.device).unsqueeze(0)

        robot_base = torch.as_tensor(obs["robot_base"][0]).to(
            self.device).unsqueeze(0)
        self.robot.write_root_pose_to_sim(robot_base, env_ids=self.env_ids)

        # extract init object setting
        target_object_name = self.env_config["params"]["Task"]["target_object"]
        object_root_pose = torch.as_tensor(
            obs[f"{target_object_name}_root_pose"][0]).to(
                self.device).unsqueeze(0)

        preset_object_root_states = self.env.scene[
            self.grasp_object].data.default_root_state[self.env_ids].clone()
        self.env.scene[self.grasp_object].write_root_pose_to_sim(
            object_root_pose, env_ids=self.env_ids)
        self.env.scene[self.grasp_object].write_root_velocity_to_sim(
            preset_object_root_states[:, 7:] * 0, env_ids=self.env_ids)
        if self.kitchen_pose_range is not None:

            self.env.scene["kitchen"].write_root_pose_to_sim(
                torch.as_tensor(obs["kitchen_pose"][0]).to(
                    self.device).unsqueeze(0), self.env_ids)

        # set the kitchen drawer to the initial state
        self.reset_kitchen_drawer(init_grasp=False if self.init_open else True)

        # reset env
        need_dof_pos = self.env.scene[
            "robot"].root_physx_view.get_dof_positions()
        need_dof_pos[:, :init_joint_pos.shape[-1]] = init_joint_pos
        self.robot.root_physx_view.set_dof_positions(need_dof_pos,
                                                     self.env_ids)

        if self.init_grasp:
            self.env_grasp.init_grasp_object_state = self.env.scene[
                self.grasp_object]._data.root_state_w[:, :3]

        for i in range(20):  # reset for stable initial statu9

            if not use_joint_pos:
                if self.use_relative_pose:
                    observation, reward, terminate, time_out, info = self.env.step(
                        torch.rand(self.env.action_space.shape,
                                   device=self.device) * 0.0)
                else:

                    observation, reward, terminate, time_out, info = self.env.step(
                        torch.as_tensor(
                            demo["obs"]["ee_pose"][0][:8]).unsqueeze(0).to(
                                self.device))
            else:

                observation, reward, terminate, time_out, info = self.env.step(
                    torch.as_tensor(
                        ["obs"]["control_joint_action"][3]).unsqueeze(0).to(
                            self.device)[..., :8])

        return observation
