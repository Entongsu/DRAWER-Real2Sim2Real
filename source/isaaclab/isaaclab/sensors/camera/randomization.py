import torch
import isaaclab.utils.math as math_utils


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)
        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return quat_list[0]
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):
            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return target_quat


def sample_camera_positions(num_samples=100,
                            radius_range=(0.8, 1.5),
                            z_min=0.7,
                            base_rot=None,
                            origin=None,
                            time_step=0):
    # Randomize the radius within the radius range
    radius = torch.rand(num_samples) * (radius_range[1] -
                                        radius_range[0]) + radius_range[0]

    # Sample spherical coordinates using PyTorch
    #theta = torch.rand(num_samples) * 2 * torch.pi  # azimuthal angle
    theta = time_step.cpu() / 180 * torch.pi * 2

    # Compute z values directly based on the radius, ensuring z is between 0.7 and the radius
    z_min = torch.full((num_samples, ), z_min)
    z_max = radius
    #z = torch.rand(num_samples) * (z_max - z_min) + z_min
    z = (torch.rand(num_samples) - 0.5) * 0.10 + z_min

    # Ensure z values are within the radius and adjust r_xy accordingly
    r_xy = torch.sqrt(torch.clamp(radius**2 - z**2, min=0.0))

    # Convert to Cartesian coordinates
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)

    # Stack the translations
    translations = torch.stack((x, y, z), dim=1)

    # Sample random Euler angles and compute rotation matrices
    yaw = torch.rand(num_samples) * 2 * torch.pi

    pitch = (torch.rand(num_samples) - 0.5) * torch.pi
    roll = (torch.rand(num_samples) - 0.5) * 2 * torch.pi

    # Create rotation matrices from Euler angles
    Rz_yaw = torch.stack([
        torch.stack(
            [torch.cos(yaw), -torch.sin(yaw),
             torch.zeros(num_samples)], dim=1),
        torch.stack([torch.sin(yaw),
                     torch.cos(yaw),
                     torch.zeros(num_samples)],
                    dim=1),
        torch.stack([
            torch.zeros(num_samples),
            torch.zeros(num_samples),
            torch.ones(num_samples)
        ],
                    dim=1)
    ],
                         dim=2)

    Ry_pitch = torch.stack([
        torch.stack(
            [torch.cos(pitch),
             torch.zeros(num_samples),
             torch.sin(pitch)],
            dim=1),
        torch.stack([
            torch.zeros(num_samples),
            torch.ones(num_samples),
            torch.zeros(num_samples)
        ],
                    dim=1),
        torch.stack(
            [-torch.sin(pitch),
             torch.zeros(num_samples),
             torch.cos(pitch)],
            dim=1)
    ],
                           dim=2)

    Rx_roll = torch.stack([
        torch.stack([
            torch.ones(num_samples),
            torch.zeros(num_samples),
            torch.zeros(num_samples)
        ],
                    dim=1),
        torch.stack(
            [torch.zeros(num_samples),
             torch.cos(roll), -torch.sin(roll)],
            dim=1),
        torch.stack(
            [torch.zeros(num_samples),
             torch.sin(roll),
             torch.cos(roll)],
            dim=1)
    ],
                          dim=2)

    # Combine rotation matrices
    rotations = Rz_yaw @ Ry_pitch @ Rx_roll
    quat = math_utils.quat_from_matrix(rotations)
    quat0 = obtain_target_quat_from_multi_angles(base_rot["axis"],
                                                 base_rot["angles"])

    quat = math_utils.quat_mul(quat,
                               quat0[None].repeat_interleave(num_samples, 0))
    translations = translations + torch.as_tensor(origin)

    return translations, quat
