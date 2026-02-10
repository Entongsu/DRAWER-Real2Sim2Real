import torch
import numpy as np
from typing import Sequence
import isaaclab.utils.math as math_utils
import warp as wp


def unproject_depth(depth: torch.Tensor,
                    intrinsics: torch.Tensor) -> torch.Tensor:
    r"""Unproject depth image into a pointcloud.

    This function converts depth images into points given the calibration matrix of the camera.

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`p_{3D}` is the 3D point, :math:`d` is the depth value, :math:`u` and :math:`v` are
    the pixel coordinates and :math:`K` is the intrinsic matrix.

    If `depth` is a batch of depth images and `intrinsics` is a single intrinsic matrix, the same
    calibration matrix is applied to all depth images in the batch.

    The function assumes that the width and height are both greater than 1. This makes the function
    deal with many possible shapes of depth images and intrinsics matrices.

    Args:
        depth: The depth measurement. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: A tensor providing camera's calibration matrix. Shape is (3, 3) or (N, 3, 3).

    Returns:
        The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()
    # check if inputs are batched
    is_batched = depth_batch.dim() == 4 or (depth_batch.dim() == 3
                                            and depth_batch.shape[-1] != 1)
    # make sure inputs are batched
    if depth_batch.dim() == 3 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)
    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]  # (H, W) -> (1, H, W)
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if depth_batch.dim() != 3:
        raise ValueError(
            f"Expected depth images to have dim = 2 or 3 or 4: got shape {depth.shape}"
        )
    if intrinsics_batch.dim() != 3:
        raise ValueError(
            f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}"
        )

    # get image height and width
    im_height, im_width = depth_batch.shape[1:]
    # create image points in homogeneous coordinates (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v],
                                             indexing="ij"),
                              dim=0).reshape(2, -1)

    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1),
                                     mode="constant",
                                     value=1.0)
    pixels = pixels.unsqueeze(0)  # (3, H x W) -> (1, 3, H x W)

    # unproject points into 3D space
    points = torch.matmul(torch.inverse(intrinsics_batch),
                          pixels)  # (N, 3, H x W)
    points = points / points[:, -1, :].unsqueeze(
        1)  # normalize by last coordinate
    # flatten depth image (N, H, W) -> (N, H x W)

    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0],
                                                       -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)

    # scale points by depth
    points_xyz = points.transpose(1, 2) * depth_batch  # (N, H x W, 3)

    # points_xyz = points_xyz.view(depth_batch.shape[0], im_height, im_width,
    #                              3).transpose_(1, 2).reshape(
    #                                  depth_batch.shape[0], -1, 3)

    # return points in same shape as input
    if not is_batched:
        points_xyz = points_xyz.squeeze(0)

    return points_xyz


def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix,
                                        dtype=torch.float32,
                                        device=device)
    if position is not None:
        position = convert_to_torch(position,
                                    dtype=torch.float32,
                                    device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation,
                                       dtype=torch.float32,
                                       device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position,
                                              orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(
            ~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)),
                                    dim=1)
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_depth_batch(
        intrinsic_matrix: torch.Tensor,
        depth: torch.Tensor,
        keep_invalid: bool = False,
        position: torch.Tensor = None,
        orientation: torch.Tensor = None,
        device: torch.device = None) -> torch.Tensor:
    """
    Creates point clouds from a batch of depth images and camera intrinsic matrices.

    Args:
        intrinsic_matrix: A (B, 3, 3) tensor providing the camera's calibration matrices.
        depth: A (B, H, W) tensor with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not.
        position: A (B, 3) tensor specifying the position of the camera in a target frame.
        orientation: A (B, 4) tensor specifying the orientation (w, x, y, z) of the camera in a target frame.
        device: The device for torch where the computation should be executed.

    Returns:
        A (B, N, 3) tensor comprising the 3D coordinates of points for each batch element.
    """

    B, H, W = depth.shape
    if device is None:
        device = depth.device

    # # Create pixel grid
    i, j = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device))
    ones = torch.ones_like(i, device=device)
    pixel_grid = torch.stack([j, i, ones], dim=-1).float()  # (H, W, 3)
    pixel_grid = pixel_grid.view(H * W, 3).T  # (3, H*W)
    pixel_grid = pixel_grid.unsqueeze(0).expand(B, -1, -1)  # (B, 3, H*W)

    # Flatten the depth map
    depth_flat = depth.view(B, 1, H * W)  # (B, 1, H*W)

    # Invert the intrinsic matrix
    intrinsic_matrix_inv = torch.inverse(intrinsic_matrix)  # (B, 3, 3)

    # Transform pixel coordinates to camera coordinates
    cam_coords = torch.bmm(intrinsic_matrix_inv, pixel_grid)  # (B, 3, H*W)
    points_camera = cam_coords * depth_flat  # (B, 3, H*W)

    # Reshape to (B, H*W, 3)
    points_camera = points_camera.permute(0, 2, 1)  # (B, H*W, 3)

    # points_camera = math_utils.unproject_depth(depth, intrinsic_matrix)
    points_camera = math_utils.transform_points(points_camera, position,
                                                orientation)

    # Remove invalid points if specified
    if not keep_invalid:
        valid_mask = ~torch.isnan(points_camera).any(
            dim=-1) & ~torch.isinf(points_camera).any(dim=-1)
        points_camera = [points_camera[i, valid_mask[i]] for i in range(B)]
        points_camera = torch.nn.utils.rnn.pad_sequence(points_camera,
                                                        batch_first=True)

    return points_camera


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quaternion: A (B, 4) tensor containing quaternions (w, x, y, z).

    Returns:
        A (B, 3, 3) tensor containing rotation matrices.
    """
    B = quaternion.size(0)
    w, x, y, z = quaternion.unbind(dim=-1)

    n = w * w + x * x + y * y + z * z
    s = 2.0 / n

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    rotation_matrix = torch.empty((B, 3, 3), device=quaternion.device)
    rotation_matrix[:, 0, 0] = 1.0 - (yy + zz)
    rotation_matrix[:, 0, 1] = xy - wz
    rotation_matrix[:, 0, 2] = xz + wy
    rotation_matrix[:, 1, 0] = xy + wz
    rotation_matrix[:, 1, 1] = 1.0 - (xx + zz)
    rotation_matrix[:, 1, 2] = yz - wx
    rotation_matrix[:, 2, 0] = xz - wy
    rotation_matrix[:, 2, 1] = yz + wx
    rotation_matrix[:, 2, 2] = 1.0 - (xx + yy)

    return rotation_matrix


def pad_last_value(sequence, max_length):
    """
    Pads the last value of each sequence to match the max_length.
    """
    last_value = sequence[-1]  # Get the last value in the sequence
    padding = last_value.unsqueeze(0).expand(max_length - sequence.size(0),
                                             -1)  # Repeat the last value
    return torch.cat([sequence, padding], dim=0)


def create_pointcloud_from_rgbd_batch(
    intrinsic_matrix: torch.Tensor,
    depth: torch.Tensor,
    rgb: torch.Tensor = None,
    normalize_rgb: bool = False,
    position: torch.Tensor = None,
    orientation: torch.Tensor = None,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """
    Creates point clouds from a batch of RGB-D images and camera intrinsic matrices.
    """

    # Retrieve XYZ point cloud
    points_xyz = create_pointcloud_from_depth_batch(intrinsic_matrix,
                                                    depth,
                                                    True,
                                                    position,
                                                    orientation,
                                                    device=device)

    B, H, W = depth.shape
    num_points = H * W

    # Handle RGB data if provided

    if rgb is not None:
        rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
        rgb = rgb[:, :, :, :3]  # (B, H, W, 3)
        points_rgb = rgb.view(B, -1, 3)  # (B, H*W, 3)

        if normalize_rgb:
            points_rgb = points_rgb.float() / 255.0

        # Concatenate RGB with XYZ to create a point cloud with color
        points_xyz_rgb = torch.cat([points_xyz, points_rgb],
                                   dim=-1)  # (B, H*W, 6)
    else:
        # If no RGB is provided, just use the XYZ point cloud
        points_xyz_rgb = points_xyz  # (B, H*W, 3)

    torch.cuda.empty_cache()

    return points_xyz_rgb


def convert_to_torch(array, dtype=torch.float32, device=None):
    """
    Converts a numpy array or wp.array to a torch tensor.

    Args:
        array: The input array to convert.
        dtype: The desired data type of the torch tensor.
        device: The device to place the tensor on.

    Returns:
        The converted torch tensor.
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(dtype=dtype, device=device)
    elif isinstance(array, torch.Tensor):
        tensor = array.to(dtype=dtype, device=device)
    else:
        raise TypeError("Unsupported array type")

    return tensor
