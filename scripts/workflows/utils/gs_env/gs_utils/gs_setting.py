import torch
from isaaclab.assets import RigidObject, Articulation
import numpy as np

import isaaclab.utils.math as math_utils
from ..gs_utils.arguments import ModelParams, PipelineParams, get_combined_args

from ..gs_utils.gaussian_model import GaussianModel
from ..gs_utils.utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from ..gs_utils.render_gs import render, Camera

from scripts.workflows.utils.gs_env.utils.gs_utils import *

import cv2


class RenderPipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    depth_ratio = 0.0
    debug = False


class GsSetting:

    def __init__(self, env_config, log_dir, device):
        self.device = device
        self.env_config = env_config
        self.log_dir = log_dir

    def init_gs(self, env):
        self.bg_color = torch.as_tensor([1, 1, 1]).to(self.device)
        self.pipe = RenderPipe()
        self.gaussians = GaussianModel(3)
        self.init_gs_camera(env)
        self.init_gs_group(env)

    def init_gs_camera(self, env):
        self.gs_cameras_setting = self.env_config["params"]["GS_Camera"]
        self.gs_cameras = []
        for gs_camera_name in self.gs_cameras_setting["cameras"].keys():
            self.gs_cameras.append(
                Camera(opegl_euler=self.gs_cameras_setting["cameras"]
                       [gs_camera_name]["euler"],
                       opengl_translate=self.gs_cameras_setting["cameras"]
                       [gs_camera_name]["translate"],
                       FoVx=self.gs_cameras_setting["FoVx"],
                       FoVy=self.gs_cameras_setting["FoVy"],
                       image_size=self.gs_cameras_setting["resolution"],
                       image_name=f"{gs_camera_name}_image",
                       zfar=self.gs_cameras_setting["zfar"],
                       znear=self.gs_cameras_setting["znear"],
                       transform_camera2world=self.
                       gs_cameras_setting["transform_camera2world"]))

    def init_gs_group(self, env):

        self.raw_gs_group_data = {}
        self.splat_objects_list = []
        splat_objects_list = self.env_config["params"]["Task"]["splat_list"]

        for gs_name in splat_objects_list:
            self.raw_gs_group_data[gs_name] = {}

            if isinstance(env.scene[gs_name], RigidObject):

                access_rigid_object_gs(
                    self.raw_gs_group_data, env.scene[gs_name], gs_name,
                    self.env_config["params"]["RigidObject"][gs_name],
                    self.device, self.log_dir)
                self.splat_objects_list.append(gs_name)
            if isinstance(env.scene[gs_name], Articulation):
                gs_link_names = access_articulation_gs(
                    self.raw_gs_group_data, env.scene[gs_name], gs_name,
                    self.env_config["params"][gs_name], self.device,
                    self.log_dir)
                self.splat_objects_list.extend(gs_link_names)

    def decrease_brightness(self, img, percent):
        value = percent / 100
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = np.maximum(0, v - value)
        final_hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return result
