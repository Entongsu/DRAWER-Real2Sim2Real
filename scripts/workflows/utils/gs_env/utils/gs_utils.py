import numpy as np

import torch

import os
from plyfile import PlyData, PlyElement
from e3nn import o3
# pip install einops==0.5.0
import einops
from einops import einsum
from e3nn import o3
import einops
import isaaclab.utils.math as math_utils


def get_transform_from_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    mat = [l.split(" ") for l in lines]
    mat = mat[:-1]
    mat = np.array(mat).astype(np.float32)
    return mat


def get_scale_from_transform(t: np.ndarray):
    return np.linalg.det(t[:3, :3])**(1 / 3)


def access_rigid_object_gs(raw_gs_group_data,
                           rigid_object,
                           rigid_object_name,
                           gs_info,
                           device,
                           log_dir=None,
                           save_ply=False):

    raw_gs_group_data[rigid_object_name] = {}

    xyz, features_dc, features_rest, opacity, scaling, rotation = load_gs_ply(
        gs_info["gs_path"], device)

    scale = gs_info["scale"][0]

    raw_gs_group_data[rigid_object_name]["xyz"] = xyz * scale
    raw_gs_group_data[rigid_object_name]["features_dc"] = features_dc
    raw_gs_group_data[rigid_object_name]["features_rest"] = features_rest
    raw_gs_group_data[rigid_object_name]["opacity"] = opacity
    raw_gs_group_data[rigid_object_name]["scaling"] = scaling + np.log(scale)
    raw_gs_group_data[rigid_object_name]["rotation"] = rotation
    if save_ply:
        save_gs_ply(f"{log_dir}/{rigid_object_name}.ply", xyz, features_dc,
                    features_rest, opacity, scaling, rotation)


def access_articulation_gs(raw_gs_group_data,
                           articulation,
                           articulation_name,
                           gs_info,
                           device,
                           log_dir=None,
                           save_ply=False):

    body_names = articulation.body_names

    features_dc_buffer = []
    features_rest_buffer = []
    opacity_buffer = []
    scaling_buffer = []
    rotation_buffer = []
    xyz_buffer = []

    gs_link_names = []

    for body_name in body_names:

        gs_link_path = gs_info["gs_path"] + "/SEGMENTED/" + f"{body_name}.ply"
        if not os.path.exists(gs_link_path):
            continue
        raw_gs_group_data[body_name] = {}
        xyz, features_dc, features_rest, opacity, scaling, rotation = load_gs_ply(
            gs_link_path, device)

        raw_gs_group_data[body_name]["xyz"] = xyz
        raw_gs_group_data[body_name]["features_dc"] = features_dc
        raw_gs_group_data[body_name]["features_rest"] = features_rest
        raw_gs_group_data[body_name]["opacity"] = opacity
        raw_gs_group_data[body_name]["scaling"] = scaling
        raw_gs_group_data[body_name]["rotation"] = rotation

        gs_link_names.append(body_name)

        if save_ply:

            features_dc_buffer.append(features_dc)
            features_rest_buffer.append(features_rest)
            opacity_buffer.append(opacity)
            scaling_buffer.append(scaling)
            rotation_buffer.append(rotation)
            xyz_buffer.append(xyz)
    if save_ply:
        all_features_dc = torch.cat(features_dc_buffer, dim=0)
        all_features_rest = torch.cat(features_rest_buffer, dim=0)
        all_opacity = torch.cat(opacity_buffer, dim=0)
        all_scaling = torch.cat(scaling_buffer, dim=0)
        all_rotation = torch.cat(rotation_buffer, dim=0)
        all_xyz = torch.cat(xyz_buffer, dim=0)
        save_gs_ply(f"{log_dir}/{articulation_name}.ply", all_xyz,
                    all_features_dc, all_features_rest, all_opacity,
                    all_scaling, all_rotation)

    return gs_link_names


def construct_list_of_attributes(
    _features_dc,
    _features_rest,
    _scaling,
    _rotation,
):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_gs_ply(path, _xyz, _features_dc, _features_rest, _opacity, _scaling,
                _rotation):

    xyz = _xyz.cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.transpose(
        1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = _features_rest.transpose(
        1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.cpu().numpy()
    scale = _scaling.cpu().numpy()
    rotation = _rotation.cpu().numpy()

    dtype_full = [(attribute, 'f4')
                  for attribute in construct_list_of_attributes(
                      _features_dc,
                      _features_rest,
                      _scaling,
                      _rotation,
                  )]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_gs_ply(path, device):
    max_sh_degree = 3
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(
        plydata.elements[0]["x"]), np.asarray(
            plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),
                   axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties
        if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1)**2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1)**2 - 1))

    scale_names = [
        p.name for p in plydata.elements[0].properties
        if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties
        if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = torch.tensor(xyz, dtype=torch.float, device=device)
    _features_dc = torch.tensor(features_dc, dtype=torch.float,
                                device=device).transpose(1, 2).contiguous()
    _features_rest = torch.tensor(features_extra,
                                  dtype=torch.float,
                                  device=device).transpose(1, 2).contiguous()
    _opacity = torch.tensor(opacities, dtype=torch.float, device=device)
    _scaling = torch.tensor(scales, dtype=torch.float, device=device)
    _rotation = torch.tensor(rots, dtype=torch.float, device=device)

    return _xyz.clone(), _features_dc.clone(), _features_rest.clone(
    ), _opacity.clone(), _scaling.clone()[..., :2], _rotation.clone()


def transform_spherical_harmonics(SH, rot_mat):
    '''
        Parameters
        ----------
        SH : torch.Tensor
            Spherical harmonics coefficients
        rot_mat : torch.Tensor
            Rotation matrix (3x3)
        Returns
        -------
        torch.Tensor
            Rotated spherical harmonics coefficients
        '''
    # print("1", SplatRenderer.mem())
    shs_feat = SH.cpu()
    rotation_matrix = rot_mat.clone().cpu()

    ## rotate shs
    # P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
    # permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix.cpu().numpy() @ P
    # rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix).to(device=shs_feat.device).float())
    rot_angles = o3._rotation.matrix_to_angles(rotation_matrix)

    # print("2", SplatRenderer.mem())
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs,
                                      'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs,
                                      'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs,
                                      'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
        D_2,
        two_degree_shs,
        "... i j, ... j -> ... i",
    )
    two_degree_shs = einops.rearrange(two_degree_shs,
                                      'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs,
                                        'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
        D_3,
        three_degree_shs,
        "... i j, ... j -> ... i",
    )
    three_degree_shs = einops.rearrange(three_degree_shs,
                                        'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    # print("3", SplatRenderer.mem())
    return shs_feat.to(rot_mat.device)


def transform_rigid_object_gs(env, gs_name, gs_info, current_gs_group_data):

    rigid_object = env.scene[gs_name]
    transformation = rigid_object.data.root_state_w[0, :7]
    xyz = gs_info["xyz"].clone()
    rotation = gs_info["rotation"].clone()
    features_rest = gs_info["features_rest"].clone()
    transformed_xyz = math_utils.transform_points(xyz, transformation[:3],
                                                  transformation[3:])

    rotation_matrix = math_utils.matrix_from_quat(rotation)
    new_rotation = math_utils.matrix_from_quat(
        transformation[3:]) @ rotation_matrix
    transformed_rotation = math_utils.quat_from_matrix(new_rotation)
    # transformed_rotation2 = math_utils.quat_mul(transformation[3:].unsqueeze(0).repeat_interleave(rotation.shape[0],dim=0), rotation)

    transformed_features_rest = transform_spherical_harmonics(
        features_rest, math_utils.matrix_from_quat(transformation[3:]))
    current_gs_group_data["xyz"].append(transformed_xyz)
    current_gs_group_data["features_dc"].append(gs_info["features_dc"])
    current_gs_group_data["features_rest"].append(transformed_features_rest)
    current_gs_group_data["opacity"].append(gs_info["opacity"])
    current_gs_group_data["scaling"].append(gs_info["scaling"])
    current_gs_group_data["rotation"].append(transformed_rotation)


def transform_articulation_gs(env, gs_name, gs_info, current_gs_group_data):
    articulation = env.scene[gs_name]
    body_names = articulation.body_names
