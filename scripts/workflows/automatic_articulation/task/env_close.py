import torch

import isaaclab.utils.math as math_utils
from scripts.workflows.automatic_articulation.utils.process_action import curobo2robot_actions, process_action


class EnvCloseCabinet:

    def __init__(
        self,
        env,
        planner,
        use_relative_pose=None,
        collision_checker=None,
        env_config=None,
    ):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.kitchen = env.scene["kitchen"]
        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker
        self.planner = planner

        self.env_config = env_config

        self.target_handle_name = self.kitchen.cfg.articulation_cfg[
            "target_drawer"]

        self.handle_id, handle_name = self.kitchen.find_bodies(
            self.target_handle_name)

        self.init_setting()

    def init_setting(self):

        self.close_speed = self.env_config["params"]["Task"]["cabinet_close"][
            "close_speed"]
        self.close_cabinet_horizon = self.env_config["params"]["Task"][
            "cabinet_close"]["close_cabinet"]

        self.close_descend = self.env_config["params"]["Task"][
            "cabinet_close"]["close_descend"]

        self.close_offset = self.env_config["params"]["Task"]["cabinet_close"][
            "close_offset"]
        self.cabinet_open_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["cabinet_close"]
            ["cabinet_open_offset"]).to(self.device)
        self.cabinet_close_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["cabinet_close"]
            ["cabinet_close_offset"]).to(self.device)
        self.close_descend_speed = torch.as_tensor(
            self.env_config["params"]["Task"]["cabinet_close"]
            ["close_descend_speed"]).to(self.device)

        curobo_ik = self.planner.curobo_ik
        init_ee_pose = torch.as_tensor(self.env_config["params"]["Task"]
                                       ["cabinet_close"]["init_ee_pos"]).to(
                                           self.device).clone()
        init_jpos = curobo_ik.plan_motion(init_ee_pose[:3].clone(),
                                          init_ee_pose[3:7].clone()).to(
                                              self.device)

        self.init_jpos = init_jpos[0]
        self.init_ee_pose = torch.cat([
            init_ee_pose.unsqueeze(0),
            torch.zeros((self.env.num_envs, 1)).to(self.device)
        ],
                                      dim=1)

        self.handle_id, handle_name = self.env.scene["kitchen"].find_bodies(
            self.target_handle_name)
        self.joint_ids, joint_names = self.kitchen.find_joints(
            self.kitchen.cfg.articulation_cfg["robot_random_range"][
                self.target_handle_name]["joint_name"])

    def reset(self, observation):

        self.get_target_handle_traj()

    def get_cabinet_open_pos(self):
        handle_location = self.kitchen._data.body_state_w[0][
            self.handle_id][:, :3]
        placement_pose = handle_location.clone()
        placement_pose[..., :3] += self.cabinet_open_offset[..., :3].unsqueeze(
            0)

        robot_dof_pos = self.robot.root_physx_view.get_dof_positions()
        placement_quat = self.cabinet_open_offset[...,
                                                  3:7].unsqueeze(0).clone()

        robot_root_pose = self.robot._data.root_state_w
        curobo_position, curobo_quat = math_utils.subtract_frame_transforms(
            robot_root_pose[:, :3], robot_root_pose[:, 3:7], placement_pose,
            placement_quat)

        return curobo_position, curobo_quat, robot_dof_pos

    def get_cabinet_close_pos(self):
        handle_location = self.kitchen._data.body_state_w[0][
            self.handle_id][:, :3]
        placement_pose = handle_location.clone()
        placement_pose[..., :3] += self.cabinet_close_offset[
            ..., :3].unsqueeze(0)

        robot_dof_pos = self.robot.root_physx_view.get_dof_positions()
        placement_quat = self.cabinet_open_offset[...,
                                                  3:7].unsqueeze(0).clone()

        robot_root_pose = self.robot._data.root_state_w

        curobo_position, curobo_quat = math_utils.subtract_frame_transforms(
            robot_root_pose[:, :3], robot_root_pose[:, 3:7], placement_pose,
            placement_quat)

        self.close_init_ee_pose, self.close_ee_quat = curobo_position, curobo_quat

        return curobo_position, curobo_quat, robot_dof_pos

    def get_target_handle_traj(self):

        curobo_position, curobo_quat, robot_dof_pos = self.get_cabinet_close_pos(
        )
        self.close_ee_pose = self.close_init_ee_pose.clone()

        self.close_ee_pose += (torch.rand(self.close_ee_pose.shape,
                                          device=self.device) - 0.5) * 0.08

        # if self.collision_checker:
        #     self.planner.add_obstacle()
        ee_pose, traj = self.planner.plan_motion(robot_dof_pos,
                                                 curobo_position, curobo_quat)

        self.planner.clear_obstacles()
        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion
        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)
        _, self.target_ee_traj = curobo2robot_actions(curobo_target_ee_pos,
                                                      self.device)

        # descend to the target pose
        descend_ee_pose = torch.cat([
            self.close_ee_pose, self.close_ee_quat,
            torch.ones((self.env.num_envs, 1)).to(self.device)
        ],
                                    dim=1).repeat(self.close_descend, 1)
        descend_ee_pose[:, 2] -= self.close_descend_speed * torch.arange(
            self.close_descend, device=self.device)
        self.target_ee_traj = torch.cat([self.target_ee_traj, descend_ee_pose],
                                        0)
        # close the cabinet
        close_actions = torch.cat([
            self.close_ee_pose, self.close_ee_quat,
            torch.ones((self.env.num_envs, 1)).to(self.device)
        ],
                                  dim=1).repeat(self.close_cabinet_horizon, 1)
        close_actions[:, 2] -= self.close_descend_speed * (self.close_descend)
        close_actions[:, 0] += self.close_speed * torch.arange(
            self.close_cabinet_horizon, device=self.device)

        self.target_ee_traj = torch.cat([self.target_ee_traj, close_actions],
                                        dim=0)
        self.reach_length = len(self.target_ee_traj)

        self.count_steps = 0

        return True

    def step(self):
        reset = False

        if self.count_steps < self.reach_length:
            target_ee_pose = self.target_ee_traj[self.count_steps].unsqueeze(0)
            actions = process_action(target_ee_pose[0], self.use_relative_pose,
                                     self.robot, self.device)

            actions[:, -1] = 1

            observation, reward, terminate, time_out, info = self.env.step(
                actions)

            self.count_steps += 1
        elif self.count_steps < self.reach_length + self.close_descend:

            if self.use_relative_pose:
                actions = torch.zeros(self.env.action_space.shape,
                                      device=self.device)
                actions[:, -1] = 1
                actions[:, 2] = 0.01
            else:

                actions = torch.cat([
                    self.close_ee_pose, self.close_ee_quat,
                    torch.ones((self.env.num_envs, 1)).to(self.device)
                ],
                                    dim=1)
                actions[:, 2] -= self.close_descend_speed * (self.count_steps -
                                                             self.reach_length)

            observation, reward, terminate, time_out, info = self.env.step(
                actions)
            self.count_steps += 1
        elif self.count_steps < self.reach_length + self.close_cabinet_horizon + self.close_descend:

            if self.use_relative_pose:
                actions = torch.zeros(self.env.action_space.shape,
                                      device=self.device)
                actions[:, -1] = 1
                actions[:, 0] = 0.05
            else:

                actions = torch.cat([
                    self.close_ee_pose, self.close_ee_quat,
                    torch.ones((self.env.num_envs, 1)).to(self.device)
                ],
                                    dim=1)
                actions[:,
                        2] -= self.close_descend_speed * (self.close_descend)
                actions[:, 0] += self.close_speed * (
                    self.count_steps - self.reach_length - self.close_descend)

            observation, reward, terminate, time_out, info = self.env.step(
                actions)
            self.count_steps += 1

            reset = self.kitchen._data.joint_pos[0][
                self.joint_ids] < self.close_offset

        else:

            return True, None, None, None, None, None, None, self.kitchen._data.joint_pos[
                0][self.joint_ids] < self.close_offset

        return reset, observation, reward, terminate, time_out, info, actions, self.kitchen._data.joint_pos[
            0][self.joint_ids] < self.close_offset

    def success_or_not(self, observation):

        close_or_not = self.kitchen._data.joint_pos[0][
            self.joint_ids] < self.close_offset

        return close_or_not
