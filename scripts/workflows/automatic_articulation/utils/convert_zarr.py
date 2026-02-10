import zarr
import numpy as np
import cprint
import h5py

import torch
from tools.visualization_utils import *

import shutil


def save_zarr(log_dir, save_dir, h5py_file):
    h5py_data = h5py.File(h5py_file, 'r+')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    num_demo = len(h5py_data["data"])
    traj_length = []
    state_arrays = []
    point_cloud_arrays = []
    action_arrays = []
    total_count = 0
    imagin_robot = []
    episode_ends_arrays = []

    for demo_id in range(0, num_demo):
        print(demo_id)
        traj_length.append(
            h5py_data["data"][f"demo_{demo_id}"]["actions"].shape[0])

        state_arrays.append(
            h5py_data["data"][f"demo_{demo_id}"]["obs"]["ee_pose"])
        sample_pc = h5py_data["data"][f"demo_{demo_id}"]["obs"]["seg_pc"][
            ..., :3]
        if "imagin_robot" in h5py_data["data"][f"demo_{demo_id}"]["obs"].keys(
        ):
            imagin_robot.append(
                h5py_data["data"][f"demo_{demo_id}"]["obs"]["imagin_robot"])
        # if "handle_points" in h5py_data["data"][f"demo_{demo_id}"]["obs"].keys(
        # ):

        #     sample_handle_points = sample_fps(
        #         h5py_data["data"][f"demo_{demo_id}"]["obs"]["handle_points"][
        #             ..., :3],
        #         num_samples=1000).cpu().numpy()
        #     sample_pc = np.concatenate([sample_pc, sample_handle_points],
        #                                axis=1)
        # cam3d = vis_pc(sample_pc[0][:, :3], None)
        # visualize_pcd([cam3d])
        point_cloud_arrays.append(sample_pc)
        action_arrays.append(h5py_data["data"][f"demo_{demo_id}"]["actions"])
        total_count += traj_length[-1]
        episode_ends_arrays.append(total_count)

    action_arrays = np.concatenate(action_arrays, axis=0)
    point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    episode_ends_arrays = np.stack(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    point_cloud_chunk_size = (traj_length[0], point_cloud_arrays.shape[1],
                              point_cloud_arrays.shape[2])

    state_chunk_size = (traj_length[0], state_arrays.shape[1])
    action_chunk_size = (traj_length[0], action_arrays.shape[1])

    zarr_data.create_dataset('state',
                             data=state_arrays,
                             chunks=state_chunk_size,
                             dtype='float32',
                             overwrite=True,
                             compressor=compressor)

    zarr_data.create_dataset('point_cloud',
                             data=point_cloud_arrays,
                             chunks=point_cloud_chunk_size,
                             dtype='float32',
                             overwrite=True,
                             compressor=compressor)

    zarr_data.create_dataset('action',
                             data=action_arrays,
                             chunks=action_chunk_size,
                             dtype='float32',
                             overwrite=True,
                             compressor=compressor)
    zarr_meta.create_dataset('episode_ends',
                             data=episode_ends_arrays,
                             dtype='int64',
                             overwrite=True,
                             compressor=compressor)

    if len(imagin_robot) > 0:

        imagin_robot = np.concatenate(imagin_robot, axis=0)
        imagin_robot_chunk_size = (imagin_robot[0].shape[1],
                                   imagin_robot.shape[1],
                                   imagin_robot.shape[2])
        zarr_data.create_dataset('imagin_robot',
                                 data=imagin_robot,
                                 chunks=imagin_robot_chunk_size,
                                 dtype='float32',
                                 overwrite=True,
                                 compressor=compressor)


from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--source_path",
                    default="",
                    help="the eval type for the training")
args_cli = parser.parse_args()
if __name__ == "__main__":
    save_zarr(log_dir=args_cli.log_dir,
              save_dir=args_cli.log_dir + f"/{args_cli.source_path}.zarr",
              h5py_file=args_cli.log_dir + f"/{args_cli.source_path}.hdf5")
