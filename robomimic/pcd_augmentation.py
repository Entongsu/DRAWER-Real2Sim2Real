import datetime
import glob
import os
import pickle

import h5py
import hydra
import numpy as np
from tqdm import trange
import trimesh
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, matrix_to_quaternion, quaternion_multiply


# Example batch_depth2xyz function
def batch_depth2xyz(depth, intrinsic_matrix):

    b, h, w = depth.shape
    ymap, xmap = np.meshgrid(np.arange(w), np.arange(h))

    x = ymap[np.newaxis, ...]
    y = xmap[np.newaxis, ...]
    z = depth

    x = (x - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (y - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]

    xyz = np.stack([x, y, z], axis=-1)
    return xyz.reshape(xyz.shape[0], -1, 3)


def batch_crop_points(points, crop_min, crop_max):
    # Extract x, y, z coordinates from points
    x_coords = points[:, :, 0]
    y_coords = points[:, :, 1]
    z_coords = points[:, :, 2]

    # Create masks for each dimension
    mask_x = np.logical_and(x_coords >= crop_min[0], x_coords <= crop_max[0])
    mask_y = np.logical_and(y_coords >= crop_min[1], y_coords <= crop_max[1])
    mask_z = np.logical_and(z_coords >= crop_min[2], z_coords <= crop_max[2])

    # Combine masks to get valid points
    valid_mask = np.logical_and.reduce((mask_x, mask_y, mask_z))

    batch_points = []
    # Apply mask to points
    for i in range(len(valid_mask)):
        new_points = points[i][valid_mask[i]]
        batch_points.append(new_points)

    return batch_point
  
def get_batch_point_cloud(depth, intrinsic_matrix, extrinsics_matrix):
    # Batch depth to point cloud conversion
    pc = batch_depth2xyz(depth, intrinsic_matrix)
    b, num, _ = pc.shape
    pc_flat = pc.reshape(b, -1, 3)

    # Append homogeneous coordinate
    pc_homo = np.concatenate([pc_flat, np.ones((b, num, 1))], axis=-1)

    # Transpose to match extrinsics_matrix shape for batch multiplication
    pc_homo_T = np.transpose(pc_homo, axes=(0, 2, 1))

    # Broadcast extrinsics_matrix to match the batch size
    extrinsics_matrix_batch = np.tile(extrinsics_matrix[np.newaxis, ...],
                                      (b, 1, 1))

    # Perform batch matrix multiplication
    transformed_pc_T = np.matmul(extrinsics_matrix_batch, pc_homo_T)

    # Transpose back and remove homogeneous coordinate
    transformed_pc = np.transpose(transformed_pc_T, axes=(0, 2, 1))[:, :, :-1]

    return transformed_pc


def batch_depth_to_pcd(cameras_dict, camera_names, depthes, crop_min,
                       crop_max):

    pcs = []
    for index, camera_name in enumerate(camera_names):
        extrinsics_matrix = cameras_dict[camera_name]["extrinsic"]
        intrinsic_matrix = cameras_dict[camera_name]["intrinsic"]

        depth_image = np.flip(depthes[index], axis=1)

        points = get_batch_point_cloud(depth_image[:, :, :, 0],
                                       intrinsic_matrix, extrinsics_matrix)
        crop_pc = batch_crop_points(points, crop_min, crop_max)
        pcs.append(crop_pc)
    batch_pc = []
    for i in range(len(pcs[0])):
        batch_pc.append(np.concatenate([pcs[0][i], pcs[1][i]], axis=0))

    return batch_pc


def pad_points(points, M):
    N = points.shape[0]
    if M < N:
        return points
        raise ValueError(f"M ({M}) should be greater than or equal to N ({N})")
    return np.pad(points, ((0, M - N), (0, 0)),
                  mode='constant',
                  constant_values=-9999)


def downsample_internal(coord, feat, num_points):
    if (coord.shape[0]) < int(num_points):
        # import IPython
        # IPython.embed()
        # visualize_points(coord)
        # print("padding points")
        coord = pad_points(coord, num_points)

    indices = np.random.choice(coord.shape[0], int(num_points), replace=False)
    coord = coord[indices]
    # feat = feat[indices]
    return coord


def batch_transform_robot_arm(xposes,
                              xquates,
                              scales,
                              meshes,
                              mesh_names,
                              downsample=None):

    all_pc = []

    xposes = np.array(xposes)[:, -len(mesh_names):]
    xquates = np.array(xquates)[:, -len(mesh_names):]

    num_vertices = 0

    for i in range(len(mesh_names)):

        xpos = torch.as_tensor(xposes[:, i]).to(device=device)

        rotation_matix = quaternion_to_matrix(
            quaternion_invert(torch.as_tensor(
                xquates[:,
                        i][:,
                           [3, 0, 1, 2]]))).to(device=device)  # xyzw -> wxyz

        transform = Transform3d(device=device).rotate(rotation_matix).scale(
            torch.as_tensor(scales * 2).to(device=device)).translate(xpos)
        mesh = meshes[mesh_names[i]].copy()
        new_vertices = transform.transform_points(
            torch.as_tensor(
                mesh.vertices,
                dtype=torch.float32).to(device=device)).cpu().numpy()

        if len(new_vertices.shape) == 2:
            new_vertices = new_vertices[None]

        if downsample is not None:
            new_vertices = batch_downsample_points(new_vertices, downsample)

        all_pc.append(new_vertices)
        num_vertices += len(mesh.vertices)

    pointcloud = np.concatenate(all_pc, axis=1)

    return pointcloud


def batch_downsample_points(point_cloud_batch, num_points=1024):
    # Get the number of points in the input point cloud batch
    # if isinstance(point_cloud_batch, np.ndarray):

    #     num_points_input = point_cloud_batch.shape[1]

    #     if num_points_input <= num_points:
    #         return point_cloud_batch  # No downsampling needed if input size is smaller than or equal to desired size

    #     # Generate random indices for downsampling
    #     indices = np.random.choice(num_points_input, num_points, replace=False)
    #     import pdb
    #     pdb.set_trace()

    #     # Extract the subset of points using the random indices
    #     downsampled_points_batch = point_cloud_batch[:, indices, :]
    # else:
    batch = len(point_cloud_batch)
    downsampled_points_batch = []
    for i in range(batch):
        downsampled_points = downsample_internal(point_cloud_batch[i],
                                                 feat=None,
                                                 num_points=num_points)
        downsampled_points_batch.append(downsampled_points)

    return np.array(downsampled_points_batch)


def batch_preprocess_pc(leftview_depthes,
                        frontview_depthes,
                        robot_arm_pos,
                        robot_arm_quat,
                        cube_pos,
                        cube_quat,
                        cube_size,
                        cube_mesh=None,
                        robot_arm_meshes=None,
                        robot_arm_names=[
                            'link0', 'link1', 'link2', 'link3', 'link4',
                            'link5', 'link6', 'link7', "eef", "leftfinger",
                            "rightfinger"
                        ],
                        cameras_dict=None,
                        downsample_robot_points=300,
                        downsample_cube_points=1000,
                        downsample_real_points=1024):

    scales = np.ones((len(robot_arm_pos), 3)) / 2
    roboot_pointcloud = batch_transform_robot_arm(
        robot_arm_pos,
        robot_arm_quat,
        scales,
        robot_arm_meshes,
        robot_arm_names,
        downsample=downsample_robot_points)

    cube_names = ["cube"]
    cube_meshes = {"cube": cube_mesh}

    cube_pointcloud = batch_transform_robot_arm(
        np.array(cube_pos)[:, None, :],
        np.array(cube_quat)[:, None, :],
        np.array(cube_size),
        cube_meshes,
        cube_names,
        downsample=downsample_cube_points)

    real_pc = batch_depth_to_pcd(cameras_dict, ["frontview", "leftview"], [
        frontview_depthes,
        leftview_depthes,
    ], [-0.1, -0.4, 0.005], [1.0, 0.4, 1.4])

    # begin downsample
    # ds_roboot_pointcloud = batch_downsample_points(
    #     roboot_pointcloud, num_points=downsample_robot_points)
    ds_real_pc = batch_downsample_points(real_pc,
                                         num_points=downsample_real_points)
    # ds_cube_pointcloud = batch_downsample_points(
    #     cube_pointcloud, num_points=downsample_cube_points)

    all_pc = np.concatenate([roboot_pointcloud, ds_real_pc, cube_pointcloud],
                            axis=1)
    downsampled_points = batch_downsample_points(
        all_pc, num_points=downsample_real_points)

    return downsampled_points


def extract_meshes_from_scene(scene):
    # This function will extract individual meshes from the scene and return them as a list of Trimesh objects
    mesh_list = []

    # Check if the scene has multiple geometries
    if len(scene.geometry) > 0:
        # Flatten the scene to a single Trimesh object if it's composed of multiple geometries
        # This combines all geometries into a single mesh
        combined_mesh = scene.dump(concatenate=True)

        # If you want to keep them separate instead, iterate over each geometry
        for mesh in scene.geometry.values():
            if isinstance(mesh, trimesh.Trimesh):
                mesh_list.append(mesh)
    else:
        # The scene is already a single mesh
        mesh_list.append(scene)
    combined_mesh = trimesh.util.concatenate(mesh_list)
    return combined_mesh


def get_robot_mesh(robot_arm_names):

    robot_arm_meshes = {}

    for name in robot_arm_names:
        mesh_path = f"/media/lme/data2/polymetis_franka/robot/real/inverse_kinematics/franka/mesh/{name}.obj"
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = extract_meshes_from_scene(mesh)

        robot_arm_meshes[name] = mesh

    return robot_arm_meshes


def augmentation_pcd(obs,
                     downsample_robot_points=300,
                     downsample_cube_points=1000,
                     downsample_real_points=1024):
    cube_mesh = trimesh.load("/media/lme/data2/weird/new_cube.stl")

    robot_arm_names = [
        'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
        "hand", "finger", "finger"
    ]
    robot_arm_meshes = get_robot_mesh(robot_arm_names)

    robot_arm_pos = obs["robot_arm_pose"][:, :, :3]
    robot_arm_quat = obs["robot_arm_pose"][:, :, 3:]

    cube_pos = obs["obj_pose"][:, :3]
    cube_quat = obs["obj_pose"][:, 3:]
    cube_size = np.ones((len(cube_quat), 3)) * 0.025

    scales = np.ones((len(robot_arm_pos), 3)) / 2
    roboot_pointcloud = batch_transform_robot_arm(
        robot_arm_pos,
        robot_arm_quat,
        scales,
        robot_arm_meshes,
        robot_arm_names,
        downsample=downsample_robot_points)

    cube_names = ["cube"]
    cube_meshes = {"cube": cube_mesh}

    cube_pointcloud = batch_transform_robot_arm(
        np.array(cube_pos)[:, None, :],
        np.array(cube_quat)[:, None, :],
        np.array(cube_size),
        cube_meshes,
        cube_names,
        downsample=downsample_cube_points)
    ds_real_pc = batch_downsample_points(obs["pcd"],
                                         num_points=downsample_real_points)

    all_pc = np.concatenate(
        [roboot_pointcloud, ds_real_pc, cube_pointcloud],  # cube_pointcloud
        axis=1)
    downsampled_points = batch_downsample_points(
        all_pc, num_points=downsample_real_points)

    return downsampled_points

