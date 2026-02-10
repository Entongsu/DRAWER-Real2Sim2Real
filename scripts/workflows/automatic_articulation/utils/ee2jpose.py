import trimesh
import h5py
import torch
import isaaclab.utils.math as math_utils
from tools.visualization_utils import *
import shutil

log_dir = 'logs/1110_close2/'
type = "close_horizon"
shutil.copy(log_dir + "close_normalized_aug.hdf5",
            log_dir + "close_normalized_aug_jpos.hdf5")
h5py_data = h5py.File(log_dir + "close_normalized_aug_jpos.hdf5", 'r+')
actions_buffer = []
for id in range(len(h5py_data['data'])):  #len(h5py_data['data'])
    print(id)

    del h5py_data["data"][f"demo_{id}"]["actions"]
    actions = np.array(
        h5py_data["data"][f"demo_{id}"]["obs"]["joint_pos"])[..., :8]
    actions_buffer.append(actions)
    # h5py_data["data"][f"demo_{id}"]["actions"] = actions

all_actions = np.concatenate(actions_buffer, axis=0)
stats = {
    "action": {
        "min": all_actions.min(axis=0),
        "max": all_actions.max(axis=0),
    }
}
np.save(log_dir + f"/{type}_stats_joint_pos.npy", stats)
for id in range(len(h5py_data['data'])):  #len(h5py_data['data'])
    print(id)
    h5py_data["data"][f"demo_{id}"]["actions"] = actions_buffer[id]
