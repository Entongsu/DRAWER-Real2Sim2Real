# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.sensors.camera.batch_utils import create_pointcloud_from_rgbd_batch, create_pointcloud_from_depth
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
"""
Root state.
"""


def control_joint_action(env: ManagerBasedRLEnv) -> torch.Tensor:

    return env.scene["robot"]._data.joint_pos_target


def robot_qpos(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    if "robot" not in env.scene.keys():
        return {}
    robot_assest = env.scene["robot"]

    robot_jpos = robot_assest.data.joint_pos

    gripper_actions = torch.sign(
        env.action_manager.get_term("gripper_action").raw_actions + 0.02)

    return torch.cat([robot_jpos[:, :7], gripper_actions], dim=1)


def ee_pose(env: ManagerBasedRLEnv, body_name="panda_hand") -> torch.Tensor:
    if "robot" not in env.scene.keys():
        return {}
    robot_assest = env.scene["robot"]
    body_id = robot_assest.find_bodies(body_name)[0][0]

    return torch.cat([
        robot_assest.data.body_pos_w[:, body_id] - env.scene.env_origins,
        robot_assest.data.body_quat_w[:, body_id]
    ],
                     dim=1)


def base_pos_z(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_b


def base_ang_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_ang_vel_b


def projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w - env.scene.env_origins


def root_quat_w(
    env: ManagerBasedEnv,
    make_quat_unique: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_link_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_w


def root_ang_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_ang_vel_w


"""
Joint state.
"""


def joint_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos  #[:, :7]
    # joint_pos = torch.cat([joint_pos[:, :7], gripper[:, None]], dim=1)

    return joint_pos


def joint_pos_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.
                                joint_ids] - asset.data.default_joint_pos[:,
                                                                          asset_cfg
                                                                          .
                                                                          joint_ids]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.
                                joint_ids] - asset.data.default_joint_vel[:,
                                                                          asset_cfg
                                                                          .
                                                                          joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv,
                sensor_cfg: SceneEntityCfg,
                offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[
        ..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv,
                         asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force(
    )[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


def imu_orientation(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")
) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_ang_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")
) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")
) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[
        sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type
            == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(
            images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get(
            "model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name",
                                              "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device",
                                                env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = [
            "resnet18", "resnet34", "resnet50", "resnet101"
        ]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(
                    self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(
                    self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data,
                                      **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str,
                                         model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}",
                                              trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406],
                                device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225],
                               device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc,
                                            interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str,
                              model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(
                models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406],
                                device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225],
                               device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


def last_action(env: ManagerBasedEnv,
                action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv,
                       command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


def process_camera_data(env: ManagerBasedRLEnv,
                        whole_rgb=False,
                        seg_rgb=False,
                        whole_pc=False,
                        seg_pc=False,
                        bbox=None,
                        segmentation_name=None,
                        max_length=None,
                        align_robot_base=False,
                        whole_depth=False) -> dict:

    if not whole_pc and not seg_pc and not whole_rgb and not seg_rgb:
        return {}

    camera_list = get_camera_data_list(env)
    if camera_list == []:
        return {}
    result = {}

    rgb_data, seg_data, intrinsic_params, depth, position, orientation, extrinsic_orientation, id2lables = extract_camera_data(
        camera_list, whole_rgb, seg_rgb, seg_pc, whole_pc, segmentation_name)

    pc_idx_per_env, num_env, num_camera = calculate_camera_indices(camera_list)

    if seg_rgb or seg_pc or whole_rgb:
        process_rgb_and_segmentation(result, rgb_data, seg_data, seg_rgb,
                                     seg_pc, whole_rgb, pc_idx_per_env,
                                     num_env, num_camera)

    if whole_pc or seg_pc:

        process_point_cloud(env, result, rgb_data, depth, intrinsic_params,
                            position, orientation, pc_idx_per_env, num_env,
                            num_camera, bbox, max_length, seg_data,
                            align_robot_base, seg_pc, whole_pc, whole_depth)

    # if seg_rgb or seg_pc or whole_pc:
    process_camera_parameters(result, position, orientation,
                              extrinsic_orientation, intrinsic_params,
                              pc_idx_per_env, num_env, num_camera)

    # torch.cuda.empty_cache()
    result["id2lables"] = id2lables

    return result


def get_camera_data_list(env):
    return [
        env.scene[name].data for name in env.scene.keys() if "camera" in name
    ]


def extract_camera_data(camera_list, whole_rgb, seg_rgb, seg_pc, whole_pc,
                        segmentation_name):
    rgb_data, seg_data, intrinsic_params, depth, position, orientation, extrinsic_orientation = [], [], [], [], [], [], []
    id2lables = []

    for data in camera_list:
        if whole_rgb or seg_rgb or seg_pc:
            rgb_data.append(data.output["rgb"])
        if seg_rgb or seg_pc:
            seg_data.append(data.output[segmentation_name])

        position.append(data.pos_w)
        orientation.append(data.quat_w_ros)
        extrinsic_orientation.append(data.quat_local_opengl)
        intrinsic_params.append(data.intrinsic_matrices)

        if whole_pc or seg_pc:

            depth.append(data.output["distance_to_image_plane"])
        if seg_pc or seg_rgb:

            if isinstance(data.info, list):
                id2lables.append(data.info[0][segmentation_name]["idToLabels"])
            else:
                id2lables.append(data.info[segmentation_name]["idToLabels"])

    return rgb_data, seg_data, intrinsic_params, depth, position, orientation, extrinsic_orientation, id2lables


def calculate_camera_indices(camera_list):
    num_env = len(camera_list[0].intrinsic_matrices)
    num_camera = len(camera_list)
    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to('cuda')
    return pc_idx_per_env[0], num_env, num_camera


def process_rgb_and_segmentation(result, rgb_data, seg_data, seg_rgb, seg_pc,
                                 whole_rgb, pc_idx_per_env, num_env,
                                 num_camera):

    rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])

    if seg_rgb or seg_pc:
        seg = torch.stack(seg_data).view(-1, *seg_data[0].shape[1:])

    rgb = rgb[pc_idx_per_env].view(num_env, num_camera,
                                   *rgb.shape[1:])[..., :3]

    if torch.max(rgb) < 1.2:
        rgb = (rgb * 255).byte()

    if seg_rgb or seg_pc:
        seg = seg[pc_idx_per_env].view(num_env, num_camera, *seg.shape[1:])
        background_mask = torch.all(seg.unsqueeze(-1) == 1, axis=-1)
        frontground_mask = ~background_mask
        front_ground = torch.zeros_like(rgb)
        front_ground[frontground_mask.squeeze(-1)] = rgb[
            frontground_mask.squeeze(-1)]

        result['seg_rgb'] = torch.cat([front_ground, seg], dim=-1)
        result['segmentation'] = seg

    if whole_rgb:
        result['rgb'] = rgb


def process_point_cloud(env,
                        result,
                        rgb_data,
                        depth,
                        intrinsic_params,
                        position,
                        orientation,
                        pc_idx_per_env,
                        num_env,
                        num_camera,
                        bbox,
                        max_length,
                        seg_data,
                        align_robot_base=False,
                        seg_pc=False,
                        whole_pc=False,
                        whole_depth=False):

    seg_mask_tensor = torch.cat(seg_data, dim=0) if seg_pc else None

    if seg_pc:
        rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])

    depth_data = torch.stack(depth).view(-1, *depth[0].shape[1:]).squeeze(-1)
    # depth_data = math_utils.convert_perspective_depth_image_to_orthogonal_depth_image(
    #     depth_data,
    #     torch.stack(intrinsic_params).view(-1, 3, 3))

    points_xyz_rgb = create_pointcloud_from_rgbd_batch(
        intrinsic_matrix=torch.stack(intrinsic_params).view(-1, 3, 3),
        depth=depth_data,
        rgb=rgb if seg_pc else None,
        position=torch.stack(position).reshape(-1, 3),
        orientation=torch.stack(orientation).reshape(-1, 4),
    )
    if whole_pc:
        result['whole_pc'] = points_xyz_rgb.view(num_env, num_camera,
                                                 *points_xyz_rgb.shape[-2:])

    # Apply segmentation mask if provided
    B, H, W = depth_data.shape

    valid_mask = (~torch.isnan(points_xyz_rgb).any(dim=-1)
                  & ~torch.isinf(points_xyz_rgb).any(dim=-1))

    if seg_mask_tensor is not None:
        if max_length > 0:
            seg_mask_flatten = seg_mask_tensor.view(
                B, -1, 1)  # Flatten the segmentation mask

            bg_mask = torch.all(seg_mask_flatten == 1,
                                dim=-1)  # Background mask
            valid_mask = ~bg_mask & valid_mask  # Invert the mask for valid points

            # Append segmentation ID to points_xyz_rgb
            points_xyz_rgb = torch.cat([points_xyz_rgb, seg_mask_flatten],
                                       dim=-1)  # (B, H*W, 4 or 7)

        else:
            seg_mask_flatten = seg_mask_tensor.view(
                B, -1, 1)  # Flatten the segmentation mask
            #Append segmentation ID to points_xyz_rgb
            points_xyz_rgb = torch.cat([points_xyz_rgb, seg_mask_flatten],
                                       dim=-1)  # (B, H*W, 4 or 7)

    # Apply bounding box filter if provided
    if align_robot_base:
        points_xyz_rgb = align_pc_to_robot_base(env, points_xyz_rgb)

    if bbox is not None:
        valid_mask = crop_point_cloud(points_xyz_rgb, bbox, valid_mask)

    valid_mask = sample_points(points_xyz_rgb, max_length, valid_mask)

    points_xyz_rgb = points_xyz_rgb[pc_idx_per_env]
    if whole_pc:
        result['whole_pc'] = points_xyz_rgb.view(num_env, num_camera,
                                                 *points_xyz_rgb.shape[-2:])

    points_xyz_rgb = points_xyz_rgb[valid_mask]

    result['seg_pc'] = points_xyz_rgb.view(num_env, num_camera,
                                           *points_xyz_rgb.shape[-2:])

    if whole_depth:

        result['depth'] = depth_data[pc_idx_per_env].view(
            num_env, num_camera, *depth_data.shape[1:])


def align_pc_to_robot_base(env, points_xyz_rgb):
    root_pose = env.scene["robot"]._data.root_state_w
    translate_root, quat_root = math_utils.subtract_frame_transforms(
        root_pose[:, :3], root_pose[:, 3:7],
        torch.zeros_like(root_pose[:, :3]),
        torch.tensor([[1., 0., 0.,
                       0.]]).to(env.device).repeat_interleave(env.num_envs, 0))

    transformed_xyz = math_utils.transform_points(
        points_xyz_rgb[..., :3],
        translate_root.repeat_interleave(len(points_xyz_rgb), 0),
        quat_root.repeat_interleave(len(points_xyz_rgb), 0))

    return torch.cat([transformed_xyz, points_xyz_rgb[..., 3:]], dim=-1)


def sample_points(points_xyz_rgb, max_length, valid_mask):

    if max_length > 0:
        valid_points_mask = valid_mask.clone()
        valid_points_mask = valid_points_mask.float(
        )  # Convert valid_mask to float for use with torch.multinomial

        # Check for batches where all valid_mask values are zero and set them to True
        all_zero_mask = valid_points_mask.sum(
            dim=1) == 0  # Identify batches with all zero valid_mask

        valid_points_mask[
            all_zero_mask] = 1.0  # Set all values to True for those batc
        # For each batch, create a valid distribution and sample from it
        valid_sample_indices = torch.multinomial(valid_points_mask,
                                                 max_length,
                                                 replacement=True)

        # Ensure valid_sample_indices is correctly shaped
        valid_sample_indices = valid_sample_indices.view(
            -1)  # Flatten if needed

        # # Set selected indices in valid_mask to True
        # valid_mask.scatter_(1, valid_sample_indices.unsqueeze(0),
        #                     True)  # Adjust dim if necessary
        valid_mask[:] = False
        valid_mask[:, valid_sample_indices] = True

    return valid_mask


def crop_point_cloud(points_xyz_rgb, bbox, valid_mask):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    bbox_mask = (points_xyz_rgb[..., 0] >= x_min) & (points_xyz_rgb[
        ..., 0] <= x_max) & (points_xyz_rgb[..., 1]
                             >= y_min) & (points_xyz_rgb[..., 1] <= y_max) & (
                                 points_xyz_rgb[..., 2]
                                 >= z_min) & (points_xyz_rgb[..., 2] <= z_max)

    valid_mask = valid_mask & bbox_mask
    return valid_mask


def process_camera_parameters(result, position, orientation,
                              extrinsic_orientation, intrinsic_params,
                              pc_idx_per_env, num_env, num_camera):
    extrinsic_position = torch.stack(position).view(-1, *position[0].shape[1:])
    extrinsic_orientation = torch.stack(extrinsic_orientation).view(
        -1, *orientation[0].shape[1:])

    extrinsic_orientation = math_utils.matrix_from_quat(extrinsic_orientation)

    extrisic_transformation = torch.eye(4).unsqueeze(0).repeat_interleave(
        len(extrinsic_position), 0).to(extrinsic_orientation.device)
    extrisic_transformation[:, :3, :3] = extrinsic_orientation
    extrisic_transformation[:, :3, 3] = extrinsic_position

    result['extrinsic_params'] = extrisic_transformation[pc_idx_per_env].view(
        num_env, num_camera, *extrisic_transformation.shape[1:3])

    intrinsic_params = torch.stack(intrinsic_params).view(
        -1, *intrinsic_params[0].shape[1:3])
    result['intrinsic_params'] = intrinsic_params[pc_idx_per_env].view(
        num_env, num_camera, *intrinsic_params.shape[1:3])


def widox250_ee_pose(env: ManagerBasedRLEnv, init_ee_pose,
                     body_name) -> torch.Tensor:
    if "robot" not in env.scene.keys():
        return {}
    robot_assest = env.scene["robot"]
    body_id = robot_assest.find_bodies(body_name)[0][0]
    ee_quat = robot_assest.data.body_quat_w[:, body_id]
    init_ee_pose = init_ee_pose.unsqueeze(
        0).repeat_interleave(  # type: ignore  
            env.num_envs, 0).to(env.device)
    delta_ee_quat = math_utils.quat_mul(
        math_utils.quat_inv(init_ee_pose[:, 3:7]), ee_quat)
    delta_ee_axis_angles = math_utils.axis_angle_from_quat(delta_ee_quat)

    return torch.cat([
        robot_assest.data.body_pos_w[:, body_id] - env.scene.env_origins,
        delta_ee_axis_angles
    ],
                     dim=1)


def target_lift_object_pos(env: ManagerBasedRLEnv, target_lift_object_pos):

    return torch.tensor(target_lift_object_pos).unsqueeze(0).repeat_interleave(
        env.num_envs, 0).to(env.device)


def get_root_state(env, name):
    root_pose = env.scene[name]._data.root_state_w[:, :7]
    root_pose[:, :3] -= env.scene.env_origins

    return root_pose


def object_ee_dist(
    env: ManagerBasedRLEnv,
    ee_frame_name: str,
    object_name: str,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    object: RigidObject = env.scene[object_name]
    ee_pos_w = env.scene[ee_frame_name].data.root_link_pos_w[:, :3]
    object_pos_w = object.data.root_link_pos_w[:, :3]
    object_pos_b = object_pos_w[:, :3] - ee_pos_w[:, :3]
    return object_pos_b


def object_tip_dist(
    env: ManagerBasedRLEnv,
    object_name: str,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    object: RigidObject = env.scene[object_name]
    wx250s_right_finger_link = env.scene[
        "wx250s_right_finger_link"].data.root_link_pos_w[:, :3]
    object_pos_w = object.data.root_link_pos_w[:, :3]
    object_wx250s_right_finger_link = object_pos_w[:, :
                                                   3] - wx250s_right_finger_link[:, :
                                                                                 3]

    wx250s_left_finger_link = env.scene[
        "wx250s_left_finger_link"].data.root_link_pos_w[:, :3]
    object_pos_w = object.data.root_link_pos_w[:, :3]
    object_wx250s_left_finger_link = object_pos_w[:, :
                                                  3] - wx250s_left_finger_link[:, :
                                                                               3]

    return torch.cat(
        [object_wx250s_right_finger_link, object_wx250s_left_finger_link],
        dim=-1)
