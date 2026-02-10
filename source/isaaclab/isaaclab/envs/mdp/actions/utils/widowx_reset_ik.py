from ..task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import torch


class WidowXResetIK:

    def __init__(self, env_cfg):
        self.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "wrist_rotate",
                "elbow",
                "shoulder",
                "forearm_roll",
                "waist",
                "wrist_angle",
                "gripper",
            ],
            body_name="wx250s_ee_gripper_link",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=False,
                                                   ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )
        self.init_robot = False
        self.env_cfg = env_cfg
        self.target_pos = torch.as_tensor(
            self.env_cfg["params"]["Task"]["init_ee_pose"], )
        self.target_joint_pos = torch.tensor([
            -1.4190e-01, 2.2390e-01, -2.8478e-01, -1.4655e-07, 1.6317e+00,
            -1.4190e-01, 1.7291e-10, 1.0000e+00, -1.0000e+00
        ],
                                             device='cuda:0')

    def reset_ee_link_pose(self, env, env_ids, target_pos=None):
        # if not self.init_robot:
        #     self.ik_controller = DifferentialInverseKinematicsAction(
        #         self.arm_action, env)
        #     self.ik_controller._asset = env.scene["robot"]
        #     # self.init_robot = True
        # target_pose = self.target_pos.clone().to(
        #     device=env.device).unsqueeze(0).repeat_interleave(env_ids.size(0),
        #                                                       dim=0)
        # self.ik_controller.process_actions(target_pose)
        # self.ik_controller.apply_actions()

        # self.ik_controller._asset.data.reset_joint_pos[
        #     ..., :7] = self.ik_controller.joint_pos_des[0].unsqueeze(
        #         0).repeat_interleave(env_ids.size(0), dim=0)
        # # for i in range(10):
        # self.ik_controller._asset.root_physx_view.set_dof_positions(
        #     self.ik_controller._asset.data.reset_joint_pos, indices=env_ids)
        # self.ik_controller._asset.root_physx_view.set_dof_velocities(
        #     self.ik_controller._asset.data.reset_joint_pos * 0.0,
        #     indices=env_ids)

        env.scene["robot"].root_physx_view.set_dof_positions(
            self.target_joint_pos.unsqueeze(0).repeat_interleave(
                env_ids.size(0), dim=0),
            indices=env_ids)
        env.scene[
            "robot"].data.reset_joint_pos = self.target_joint_pos.unsqueeze(
                0).repeat_interleave(env_ids.size(0), dim=0)
