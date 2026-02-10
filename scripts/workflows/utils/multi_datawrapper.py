import torch

from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test, fps_points
import json
import h5py
import os
import shutil
import numpy as np

# from dgl.geometry import farthest_point_sampler
import isaaclab.utils.math as math_utils
import copy
import imageio
import gzip


class MultiDatawrapper:

    def __init__(self,
                 args_cli,
                 env_config,
                 filter_keys=[],
                 use_fps=False,
                 load_path=None,
                 save_path=None,
                 train_percentate=0.9,
                 normalize_action=True,
                 use_joint_pos=False,
                 save_npz=True,
                 load_normalize_action=False,
                 save_zip=False):

        self.args_cli = args_cli

        self.env_config = env_config
        self.filter_keys = filter_keys

        self.train_percentate = train_percentate

        self.use_fps = use_fps

        self.save_npz = save_npz
        self.traj_count = 0
        self.use_joint_pos = use_joint_pos
        self.save_zip = save_zip

        if save_path is not None:
            self.save_path = self.args_cli.log_dir + "/" + save_path
            os.makedirs(self.save_path, exist_ok=True)

            self.traj_count = len(os.listdir(self.save_path))

        if load_path is not None:
            self.load_path = self.args_cli.log_dir + f"/{load_path}.hdf5"
            self.load_h5py()
            if save_path is not None and normalize_action:
                self.normalize_h5py()

        if load_normalize_action:
            self.load_normalization_stats()

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(self.args_cli.task,
                                                     self.args_cli.log_dir,
                                                     filename, num_demos)
        collector_interface.reset()

        save_config_json = json.dumps(self.env_config)
        collector_interface._h5_data_group.attrs[
            "env_setting"] = save_config_json
        return collector_interface

    def init_collector_interface(self):

        if not self.save_npz:

            self.collector_interface = self.init_collectors(
                self.args_cli.num_demos, filename=self.save_path)
        else:
            os.makedirs(self.save_path, exist_ok=True)

    def load_h5py(self):

        self.raw_data = h5py.File(f"{self.load_path}", 'r+')

    def add_demonstraions_to_buffer(self,
                                    obs_buffer,
                                    actions_buffer,
                                    rewards_buffer,
                                    does_buffer,
                                    next_obs_buffer=None):
        stop = False

        if not self.save_npz:
            stop = self.save_to_h5py(obs_buffer, actions_buffer,
                                     rewards_buffer, does_buffer,
                                     next_obs_buffer)

        else:
            self.save_to_npz(
                obs_buffer,
                actions_buffer,
                rewards_buffer,
                does_buffer,
                next_obs_buffer,
            )

            if self.traj_count == (self.args_cli.num_demos - 1):
                stop = True
            if self.traj_count == 0:
                np.save(f"{self.save_path}/env_setting.npy", self.env_config)
        self.traj_count += 1
        return stop

    def downsample_points(self, obs_buffer):
        handle_points_buffer = []
        pc_buffer = []

        if "seg_pc" in obs_buffer[0].keys():
            for index in range(len(obs_buffer)):
                obs = obs_buffer[index]
                if "seg_pc" in obs.keys():
                    points = obs["seg_pc"]

                    pc_buffer.append(points)
                if "handle_points" in obs.keys():
                    handle_points = obs["handle_points"]
                    handle_points_buffer.append(handle_points)
            point_clouds = torch.cat(pc_buffer, dim=0)
            sample_points = fps_points(point_clouds)
            if "handle_points" in obs.keys():
                handle_points_buffer = torch.cat(handle_points_buffer, dim=0)
                sample_points = torch.cat(
                    [sample_points, handle_points_buffer], dim=1)
            print(sample_points.size())
        return sample_points

    def filter_obs_buffer(self, obs_buffer):

        save_obs_buffer = []

        if self.use_fps:
            if "seg_pc" in obs_buffer[0].keys():
                if "seg_pc" not in self.filter_keys:
                    sample_points = self.downsample_points(obs_buffer)

        for index, obs in enumerate(obs_buffer):

            per_obs = {}

            for keys in obs.keys():
                if keys in self.filter_keys:
                    continue

                if "seg_pc" in keys:

                    per_obs[keys] = sample_points[index][..., :6].unsqueeze(0)

                else:
                    per_obs[keys] = obs[keys]

            save_obs_buffer.append(per_obs)
        return save_obs_buffer

    def save_to_npz(self, obs_buffer, actions_buffer, rewards_buffer,
                    does_buffer, next_obs_buffer):
        filename = os.path.join(self.save_path,
                                f"episode_{self.traj_count}.npz")

        save_obs_buffer = self.filter_obs_buffer(obs_buffer)
        if next_obs_buffer is not None:
            save_next_obs_buffer = self.filter_obs_buffer(next_obs_buffer)
        else:
            save_next_obs_buffer = None

        data = {
            'obs': save_obs_buffer,
            'actions': actions_buffer,
            'rewards': rewards_buffer,
            'dones': does_buffer,
            "next_obs": save_next_obs_buffer
        }
        if self.save_zip:
            with gzip.open(f"{filename}.gz", "wb") as f:
                torch.save(data, f, pickle_protocol=4)
        else:
            torch.save(data, filename)
        print(f"Saved episode {self.traj_count} to {filename}", "length",
              len(save_obs_buffer))

    def save_to_h5py(self, obs_buffer, actions_buffer, rewards_buffer,
                     does_buffer, next_obs_buffer):

        if self.collector_interface._is_stop:
            return True

        if self.use_fps:
            if "seg_pc" in obs_buffer[0].keys():
                sample_points = self.downsample_points(obs_buffer)

        for index in range(len(obs_buffer)):
            obs = obs_buffer[index]
            rewards = rewards_buffer[index]
            dones = does_buffer[index]
            if index == len(obs_buffer) - 1:
                dones[:] = torch.tensor([True], device='cuda:0')
            else:
                dones[:] = torch.tensor([False], device='cuda:0')

            for key, value in obs.items():
                if key in self.filter_keys:
                    continue
                if "seg_pc" in key:

                    if self.use_fps:  # collect data mode for training

                        self.collector_interface.add(
                            f"obs/{key}", sample_points[index].unsqueeze(0))
                    else:  # eval mode
                        self.collector_interface.add(f"obs/{key}", value)

                else:
                    self.collector_interface.add(f"obs/{key}", value)
            self.collector_interface.add("actions", actions_buffer[index])
            self.collector_interface.add("rewards", rewards)

            self.collector_interface.add("dones", dones)

            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

            self.collector_interface.flush(reset_env_ids)
        torch.cuda.empty_cache()
        return False

    def load_normalization_stats(self):
        if not self.use_joint_pos:
            self.action_stats = np.load(self.args_cli.log_dir + f"/stats.npy",
                                        allow_pickle=True).item()
        else:
            self.action_stats = np.load(self.args_cli.log_dir +
                                        f"/stats_joint_pos.npy",
                                        allow_pickle=True).item()

    def normalize_h5py(self):

        self.action_stats, self.raw_data = self.normalize_ations(self.raw_data)

    def normalize_ations(self, data):

        actions_buffer = []

        if not self.use_joint_pos:
            for demo_id in range(len(data["data"].keys())):

                actions = data["data"][f"demo_{demo_id}"]["actions"]
                actions_buffer.append(actions)

            all_actions = np.concatenate(actions_buffer, axis=0)[..., :3]

            stats = {
                "action": {
                    "min": all_actions.min(axis=0),
                    "max": all_actions.max(axis=0),
                }
            }
            # Save stats to a separate file
            np.save(self.args_cli.log_dir + f"/stats.npy", stats)
        else:
            for demo_id in range(len(data["data"].keys())):

                actions = data["data"][f"demo_{demo_id}"]["obs"][
                    "control_joint_action"]
                actions_buffer.append(actions)
            all_actions = np.concatenate(actions_buffer, axis=0)[..., :8]

            all_actions[:, -1] = np.sign(all_actions[:, -1] - 0.01)

            stats = {
                "action": {
                    "min": all_actions.min(axis=0),
                    "max": all_actions.max(axis=0),
                }
            }
            np.save(self.args_cli.log_dir + f"/stats_joint_pos.npy", stats)

        # # Normalize actions for each demo and save them to the copied HDF5 file
        # for demo_id in range(len(data["data"].keys())):
        #     actions = data["data"][f"demo_{demo_id}"]["actions"]

        #     # Normalize the actions using the calculated stats
        #     actions_buffer = self.normalize(actions, stats["action"])

        #     data["data"][f"demo_{demo_id}"].create_dataset("actions",
        #                                                    data=actions_buffer)

        return stats, data

    def normalize(self, arr, stats):
        min_val, max_val = stats["min"], stats["max"]
        return 2 * (arr - min_val) / (max_val - min_val) - 1

    def unnormalize(self, arr, stats):

        min_val, max_val = stats["min"], stats["max"]

        if isinstance(arr, torch.Tensor):
            max_val = torch.tensor(max_val, device=arr.device)
            min_val = torch.tensor(min_val, device=arr.device)

            result = (0.5 * (arr + 1) * (max_val - min_val) + min_val)[0]
        else:
            result = 0.5 * (arr + 1) * (max_val - min_val) + min_val

        return result

    def split_set(self):

        h5_file = h5py.File(f"{self.save_path}", 'a')
        sample_train_test(h5_file)

    def save_video(self, video_name, image_buffer):
        video_writer = imageio.get_writer(f"{video_name}.mp4", fps=10)

        for image in image_buffer:
            video_writer.append_data(image)
        video_writer.close()
