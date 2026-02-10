from robomimic.utils.test_utils import test_eval_agent_from_checkpoint
import robomimic.utils.file_utils as FileUtils

import pdb

import argparse
import json

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *
import cv2
from robosuite.utils.camera_utils import *
from robosuite.utils.visualization_utils import vis_pc
import pdb
from robosuite.utils.transform_utils import *
from robosuite.controllers.curobo_planner import *
import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml
import cv2

from robosuite.utils.mesh_utils import *

downsampled_points = np.load("test12.npy")

robot_o3d = o3d.geometry.PointCloud()
robot_o3d.points = o3d.utility.Vector3dVector(downsampled_points)
colros = np.zeros((len(downsampled_points), 3))
colros[:int(len(downsampled_points) / 2)] = [1, 0, 0]
colros[int(len(downsampled_points) / 2):] = [0, 0, 1]
robot_o3d.colors = o3d.utility.Vector3dVector(colros)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([robot_o3d,coordinate_frame])

file = h5py.File(
    "/media/lme/data2/weird/training/h5py/demo_lift_pcd_speed1024.h5py", 'r')

dataset = file["data"]
num_traj = len(dataset.keys())
demonstration = dataset[f"demo_{10}"]
obs = demonstration["obs"]
writer = imageio.get_writer("test.mp4")
for i in range(len(obs["pcd"])):
    # visualize_pc(obs["pcd"][i])

    visualize_pcd(obs["pcd"][i],
                  video_writer=writer,
                  addtional_coornage=obs["robot0_eef_pos"][i],
                  addtional_quat=None)
