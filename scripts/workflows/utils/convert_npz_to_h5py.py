from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import pickle
import json

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# # # launch omniverse app

from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test
import json
import h5py
import os
import shutil
import numpy as np
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

args_cli = parser.parse_args()
import torch


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy floats to Python floats
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)  # Convert NumPy ints to Python ints
    return obj


class Converter:

    def __init__(
        self,
        args_cli,
        filter_keys,
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path

        self.end_name = args_cli.end_name
        self.add_noise = args_cli.noise_pc
        self.filter_keys = filter_keys

        if self.add_noise:
            self.noise_level = 2.0
            self.noise_name = "_noise"
        else:
            self.noise_level = 0
            self.noise_name = ""

        if "IK" not in args_cli.task:
            self.end_name += "_joint_pos"
            # self.noise_name += "_joint_pos"

        self.init_collector_interface()

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(
            self.args_cli.task, os.path.dirname(self.save_path), filename,
            num_demos)
        collector_interface.reset()

        return collector_interface

    def init_collector_interface(self):

        self.collector_interface = self.init_collectors(
            self.args_cli.num_demos,
            filename=self.save_path.split("/")[-1] + self.end_name +
            self.noise_name)

    def filter_obs_buffer(self, obs_buffer, index, obs_name="obs"):
        if "policy" in obs_buffer[index]:
            obs = obs_buffer[index]["policy"]
        else:
            obs = obs_buffer[index]

        for key, value in obs.items():
            if key in self.filter_keys:
                continue

            if key == "seg_pc":

                # value = value[:, :1024]
                multiplicative_noise = 1 + torch.randn(
                    value.shape[1], 1) * 0.01 * self.noise_level

                value[..., :3] = value[..., :3] * multiplicative_noise.to(
                    value.device)
            if key == "joint_pos":

                value[:, -2:] = torch.sign(value[:, -2:])

            if self.add_noise:
                if key == "ee_pose":

                    noise = torch.normal(mean=0.0, std=0.0, size=value.shape)
                    value = value + noise.to(value.device)

                    value = torch.cat(
                        [value,
                         torch.as_tensor([[index]]).to(value.device)],
                        dim=1)
            if key in ["language_intruction", "failure_reasoning"]:

                for lang_index in range(len(value)):
                    value[lang_index] = value[lang_index].encode("utf-8")

            self.collector_interface.add(f"{obs_name}/{key}", value)

    def convert(self):

        self.env_config = np.load(f"{self.load_path}/env_setting.npy",
                                  allow_pickle=True)

        config = self.env_config.item()

        # Use the `default` parameter of json.dumps to handle non-serializable objects
        json_string = json.dumps(config,
                                 default=convert_to_serializable,
                                 indent=4)
        self.collector_interface._h5_data_group.attrs[
            "env_setting"] = json_string

        for file in os.listdir(f"{self.load_path}{self.end_name}"):
            if file.endswith(".npz"):

                data = torch.load(os.path.join(
                    f"{self.load_path}{self.end_name}", file),
                                  pickle_module=pickle)

                obs_buffer = data["obs"]
                actions_buffer = data["actions"]

                rewards_buffer = data["rewards"]
                does_buffer = data["dones"]

                for index in range(len(obs_buffer)):
                    self.filter_obs_buffer(obs_buffer, index, obs_name="obs")
                    # if "next_obs" in data.keys():
                    #     self.filter_obs_buffer(data["next_obs"],
                    #                            index,
                    #                            obs_name="next_obs")
                    rewards = rewards_buffer[index]
                    dones = does_buffer[index]
                    if index == len(obs_buffer) - 1:
                        dones[:] = torch.tensor([True], device='cuda:0')
                    else:
                        dones[:] = torch.tensor([False], device='cuda:0')

                    self.collector_interface.add(
                        "actions", actions_buffer[index].unsqueeze(0))

                    self.collector_interface.add("rewards", rewards)

                    self.collector_interface.add("dones", dones)

                    reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

                    self.collector_interface.flush(reset_env_ids)
                    torch.cuda.empty_cache()
        file_name = self.save_path.split("/")[-1]

        sample_train_test(
            h5py.File(
                os.path.dirname(self.save_path) + f"/{file_name}" +
                self.end_name + self.noise_name + ".hdf5", 'r+'))

        return False


Converter(
    args_cli,
    filter_keys=["segmentation", "seg_rgb", "id2lables", "seg_pc"],
).convert()
