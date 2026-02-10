#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .utils.sh_utils import eval_sh
from .utils.point_utils import depth_to_normal
from torch import nn
import numpy as np
from .utils.graphics_utils import getWorld2View2, getProjectionMatrix
import isaaclab.utils.math as math_utils

from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles


class Camera(nn.Module):

    def __init__(self,
                 opegl_euler,
                 opengl_translate,
                 FoVx,
                 FoVy,
                 image_size,
                 image_name,
                 zfar=100.0,
                 znear=0.01,
                 transform_camera2world=True,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0,
                 data_device="cuda"):
        super(Camera, self).__init__()

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.colmap_id = 0
        self.gt_alpha_mask = None
        self.uid = 123

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.image_width = image_size[0]
        self.image_height = image_size[1]

        self.zfar = zfar
        self.znear = 0.0001

        self.trans = trans
        self.scale = scale
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
            fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        R, T = self.init_camera_pose(opegl_euler, opengl_translate,
                                     transform_camera2world)
        self.set_extrinsics(R, T)

    def init_camera_pose(self, euler, translate, transform_camera2world=True):
        euler = torch.as_tensor([euler]) / 180.0 * torch.pi

        camera_rot = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1],
                                                    euler[:, 2])[0]
        if transform_camera2world:
            camera_wolrd_quat = math_utils.convert_camera_frame_orientation_convention(
                camera_rot, origin="opengl", target="world")
        else:

            camera_wolrd_quat = camera_rot.clone()

        camera_translate = translate

        p_mat = torch.as_tensor([0.5000, -0.5000, 0.5000, -0.5000])

        camera_wolrd_quat_proj = math_utils.quat_mul(camera_wolrd_quat, p_mat)
        rotate = math_utils.matrix_from_quat(camera_wolrd_quat_proj)

        return rotate.cpu().numpy(), np.array(camera_translate)

    # def init_camera_pose(self, euler, translate, transform_camera2world=True):
    #     from scipy.spatial.transform import Rotation as R
    #     cam_rot = R.from_euler('xyz', euler, degrees=True).as_matrix()
    #     p_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    #     cam_r = cam_rot @ p_mat

    #     return cam_r, np.array(translate)

    def set_extrinsics(self, R, T):
        self.R = R
        self.T = T
        test = np.zeros(3)
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, test, T,
                           self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def render(viewpoint_camera,
           pc,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color.to(torch.float32),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
        near_n=viewpoint_camera.znear,
        far_n=viewpoint_camera.zfar,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                                [0, H / 2, 0, (H - 1) / 2],
                                [0, 0, far - near, near],
                                [0, 0, 0, 1]]).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(
                0, 2, 1).reshape(-1, 9)  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(
        1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(
            2, 0, 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (
        pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return rets
