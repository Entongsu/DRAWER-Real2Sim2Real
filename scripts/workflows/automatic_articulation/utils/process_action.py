import h5py

import numpy as np

import torch

import shutil

import isaaclab.utils.math as math_utils
import os


def curobo2robot_actions(curobo_target_ee_pos, device, close_gripper=False):

    reach_length = len(curobo_target_ee_pos)
    if close_gripper:
        gripper_value = -1
    else:
        gripper_value = 1

    target_ee_traj = torch.cat([
        curobo_target_ee_pos[:, :3], curobo_target_ee_pos[:, 3:7],
        torch.ones(reach_length, 1).to(device) * gripper_value
    ],
                               dim=1)
    return reach_length, target_ee_traj


def process_action(target_ee_pose, use_relative_pose, robot, device):
    if use_relative_pose:
        ee_pose, ee_quat = get_robottip_pose(robot,
                                             device,
                                             use_gripper_offset=False)

        delta_pos = target_ee_pose[:, :3] - ee_pose

        ee_quat01 = target_ee_pose[:, 3:7]
        delta_quat = math_utils.quat_mul(ee_quat01,
                                         math_utils.quat_inv(ee_quat))

        # delta_pos, delta_quat = math_utils.subtract_frame_transforms(
        #     target_ee_pose, ee_quat_b, target_ee_pose[:, 0:3], target_ee_pose[:,
        #                                                                  3:7])
        delta_euler = math_utils.euler_xyz_from_quat(delta_quat)
        detla_euler_action = torch.cat([
            delta_euler[0].reshape(-1, 1), delta_euler[1].reshape(-1, 1),
            delta_euler[2].reshape(-1, 1)
        ],
                                       dim=1)

        actions = torch.cat([
            delta_pos, detla_euler_action, target_ee_pose[:, -1].unsqueeze(0)
        ],
                            dim=1)
    else:
        actions = target_ee_pose.unsqueeze(0)
    return actions


def reset_env(env, actions, robot, joint_pos, robot_base=None):

    last_obs, _ = env.reset()
    if robot_base is not None:
        env_ids = torch.arange(env.num_envs).to(env.device)
        root_states = robot.data.default_root_state[env_ids].clone()

        robot.write_root_pose_to_sim(robot_base, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_states[:, 7:] * 0,
                                         env_ids=env_ids)
    for i in range(10):
        last_obs, reward, terminate, time_out, info = env.step(
            torch.as_tensor(actions[0]).unsqueeze(0).to(env.device) * 0.0)
        robot.root_physx_view.set_dof_positions(
            joint_pos, indices=torch.arange(env.num_envs).to(env.device))
        robot.root_physx_view.set_dof_velocities(
            joint_pos * 0.0, indices=torch.arange(env.num_envs).to(env.device))
    # print(joint_pos)
    # last_obs, _ = env.reset()
    return last_obs


def recover_abs_actions(ee_pos, ee_quat, delta_pose):
    delta_quat = math_utils.quat_from_euler_xyz(delta_pose[:, 3],
                                                delta_pose[:, 4],
                                                delta_pose[:, 5])

    target_pos, target_rot = math_utils.combine_frame_transforms(
        ee_pos, ee_quat, delta_pose[:, 0:3], delta_quat)
    return target_pos, target_rot


def get_robottip_pose(robot, device, use_gripper_offset=True):
    # obtain quantities from simulation
    ee_pose_w = robot.data.body_state_w[:, 8, :7]
    root_pose_w = robot.data.root_state_w[:, :7]
    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7])
    # account for the offset
    if use_gripper_offset:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b,
            torch.tensor([[0.0000, 0.0000, 0.1070]], device=device),
            torch.tensor([[1., 0., 0., 0.]], device=device))
    return ee_pose_b, ee_quat_b


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


def unnormalize(arr, stats, scale=100):

    min_val, max_val = stats["min"], stats["max"]
    arr = np.array(arr)
    result = 0.5 * (arr + 1) * (max_val - min_val) + min_val
    result[..., :-1] /= scale
    return result


def normalize_action(args_cli, h5py_file_path, scale):
    # Create a copy of the original HDF5 file
    copied_file_path = args_cli.log_dir + "/normalize.hdf5"
    if os.path.exists(copied_file_path):
        os.remove(copied_file_path)
    shutil.copy(h5py_file_path, copied_file_path)

    # Open the copied file for modification
    with h5py.File(copied_file_path, 'r+') as h5py_file:
        actions_buffer = []

        # Concatenate all actions from all demos
        for demo_id in range(len(h5py_file["data"].keys())):

            h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] = np.sign(
                h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] + 0.2)
            actions = h5py_file["data"][f"demo_{demo_id}"]["actions"]
            actions_buffer.append(actions)

        all_actions = np.concatenate(actions_buffer, axis=0)
        all_actions[:, :-1] *= scale

        # Calculate min and max for normalization
        stats = {
            "action": {
                "min": all_actions.min(),
                "max": all_actions.max(),
            }
        }

        # Save stats to a separate file
        np.save(args_cli.log_dir + "/stats.npy", stats)

        # Normalize actions for each demo and save them to the copied HDF5 file
        for demo_id in range(len(h5py_file["data"].keys())):
            actions = h5py_file["data"][f"demo_{demo_id}"]["actions"]
            actions[..., :-1] *= scale

            # Normalize the actions using the calculated stats
            actions_buffer = normalize(actions, stats["action"])

            # Delete the existing dataset and replace with the normalized actions
            del h5py_file["data"][f"demo_{demo_id}"]["actions"]
            h5py_file["data"][f"demo_{demo_id}"].create_dataset(
                "actions", data=actions_buffer)

    return copied_file_path


def add_demonstraions_to_buffer(collector_interface, obs_buffer,
                                actions_buffer, rewards_buffer, does_buffer):
    for index in range(len(obs_buffer)):
        obs = obs_buffer[index]
        rewards = rewards_buffer[index]
        dones = does_buffer[index]
        if index == len(obs_buffer) - 1:
            dones[:] = torch.tensor([True], device='cuda:0')
        else:
            dones[:] = torch.tensor([False], device='cuda:0')

        for key, value in obs.items():

            collector_interface.add(f"obs/{key}", value)
        collector_interface.add("actions", actions_buffer[index])
        collector_interface.add("rewards", rewards)

        collector_interface.add("dones", dones)
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        collector_interface.flush(reset_env_ids)
