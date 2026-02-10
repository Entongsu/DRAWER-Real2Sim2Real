import trimesh
import h5py
import torch
import isaaclab.utils.math as math_utils
from tools.visualization_utils import *
import shutil
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--source_path",
                    default="",
                    help="the eval type for the training")

args_cli = parser.parse_args()
log_dir = args_cli.log_dir
shutil.copy(log_dir + "/" + args_cli.source_path + ".hdf5",
            log_dir + "/" + args_cli.source_path + "_aug.hdf5")
h5py_data = h5py.File(log_dir + "/" + args_cli.source_path + "_aug.hdf5", 'r+')
mesh_dir = load_robot_mesh(log_dir)
for id in range(len(h5py_data['data'])):  #len(h5py_data['data'])
    print(id)
    robot_links = []

    robot_link_pose = torch.as_tensor(
        h5py_data["data"][f"demo_{id}"]["obs"]["robot_link_pose"])

    imagin_robot = aug_robot_mesh(robot_link_pose,
                                  mesh_dir=mesh_dir,
                                  sample_robot_poins=400)

    if "imagin_robot" in h5py_data["data"][f"demo_{id}"]["obs"].keys():
        del h5py_data["data"][f"demo_{id}"]["obs"]["imagin_robot"]

    h5py_data["data"][f"demo_{id}"]["obs"]["imagin_robot"] = imagin_robot
