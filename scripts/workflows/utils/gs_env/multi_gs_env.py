from .gs_utils.gaussian_model import GaussianModel
from .gs_utils.utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from .gs_utils.render_gs import render, Camera

from argparse import ArgumentParser

import torch

from isaaclab.assets import RigidObject, Articulation
import numpy as np

import isaaclab.utils.math as math_utils
import os
from scripts.workflows.utils.gs_env.utils.gs_utils import *
from isaaclab.envs import ManagerBasedEnv
from .gs_utils.gs_setting import GsSetting


class MultiGaussianEnv(GsSetting):

    def __init__(self, env_config, log_dir="logs/", device="cuda"):

        self.device = device
        self.env_config = env_config
        self.log_dir = log_dir
        self.env = None
        self.initialize_gs = False

        super().__init__(env_config, log_dir, device)

    def extract_gaussians(self, env: ManagerBasedEnv):
        if not self.initialize_gs:
            self.init_gs(env)
            self.initialize_gs = True
        current_gs_group_data = {
            "xyz": [],
            "features_dc": [],
            "features_rest": [],
            "opacity": [],
            "scaling": [],
            "rotation": []
        }

        for gs_name in self.splat_objects_list:

            if isinstance(env.scene[gs_name], RigidObject):
                transform_rigid_object_gs(env, gs_name,
                                          self.raw_gs_group_data[gs_name],
                                          current_gs_group_data)
            if isinstance(env.scene[gs_name], Articulation):

                transform_articulation_gs(env, gs_name,
                                          self.raw_gs_group_data[gs_name],
                                          current_gs_group_data)
        _xyz = torch.cat(current_gs_group_data["xyz"], dim=0)
        _features_dc = torch.cat(current_gs_group_data["features_dc"], dim=0)
        _features_rest = torch.cat(current_gs_group_data["features_rest"],
                                   dim=0)
        _opacity = torch.cat(current_gs_group_data["opacity"], dim=0)

        _scaling = torch.cat(current_gs_group_data["scaling"], dim=0)
        _rotation = torch.cat(current_gs_group_data["rotation"], dim=0)
        # save_gs_ply(self.log_dir + "/realtime_splat.ply", _xyz, _features_dc,
        #             _features_rest, _opacity, _scaling, _rotation)
        # import pdb
        # pdb.set_trace()

        self.gaussians._xyz = _xyz
        self.gaussians._features_dc = _features_dc
        self.gaussians._features_rest = _features_rest
        self.gaussians._opacity = _opacity
        self.gaussians._scaling = _scaling
        self.gaussians._rotation = _rotation
        gs_images = []
        # import time
        # start = time.time()
        for index, camera in enumerate(self.gs_cameras):
            render_pkg = render(camera, self.gaussians, self.pipe,
                                self.bg_color)

            image = render_pkg["render"]
            image = image.permute(1, 2, 0).clone()

            rgb = torch.as_tensor(image * 255, dtype=torch.uint8)  #.flip(-1)
            # # [..., ::-1]
            # import cv2
            # cv2.imwrite(self.log_dir + f"/gs_image_{index}.png",
            #             rgb.detach().cpu().numpy()[:, :, ::-1])
            gs_images.append(rgb)
        # print(f"Time taken for rendering: {time.time() - start }")
        gs_images = torch.stack(gs_images, dim=0)

        return gs_images.unsqueeze(0)
