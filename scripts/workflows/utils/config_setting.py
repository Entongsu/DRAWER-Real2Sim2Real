from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg, AssetBaseCfg, ArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg, ParticleBodyPropertiesCfg
from isaaclab.sim.spawners import UsdFileCfg
import isaaclab.utils.math as math_utils
import torch

from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
from isaaclab.utils.math import quat_from_matrix
from isaaclab.sensors.camera.utils import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import json
import numpy as np

from isaaclab.sensors.ray_caster import RayCasterCfg, patterns


def get_transform_from_txt(txt_path: str):
    with open(txt_path, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    mat = [l.split(" ") for l in lines]
    mat = mat[:-1]
    mat = np.array(mat).astype(np.float32)
    return mat


def get_scale_from_transform(t: np.ndarray):
    return np.linalg.det(t[:3, :3])**(1 / 3)


def get_parameters_from_intrinsics_matrix(
    camera_matrix,
    width,
    height,
    pixel_size=3 * 1e-3,
):
    ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
    horizontal_aperture = pixel_size * width  # The aperture size in mm
    vertical_aperture = pixel_size * height
    focal_length_x = fx * pixel_size
    focal_length_y = fy * pixel_size
    focal_length = (focal_length_x +
                    focal_length_y) / 2  # The focal length in mm
    return focal_length, horizontal_aperture, vertical_aperture


def load_camera_params(camera_name, camera_setting, camera_json):

    if camera_json is not None:

        with open(camera_json, 'r') as file:
            camera_data = json.load(file)
        for cam in camera_data:
            if cam["camera_name"] == camera_name:
                camera_cfg = cam
                break

        camera_orient = torch.tensor(camera_cfg["camera_base_ori"])

        transformation_matrix = torch.tensor([[1.0, 0.0,
                                               0.0], [0.0, -1.0, 0.0],
                                              [0.0, 0.0, -1.0]])

        # Perform matrix multiplication
        camera_orient = camera_orient @ transformation_matrix
        camera_rot = quat_from_matrix(camera_orient)
        camera_translate = list(
            torch.as_tensor(camera_cfg["camera_base_pos"]).reshape(-1).numpy())

        intrinsics = camera_cfg["intrinsics"]
        K = torch.as_tensor([[intrinsics['fx'], 0, intrinsics['ppx']],
                             [0, intrinsics['fy'], intrinsics['ppy']],
                             [0, 0, 1]])

        focal_length, horizontal_aperture, vertical_aperture = get_parameters_from_intrinsics_matrix(
            camera_matrix=K,
            width=intrinsics["width"],
            height=intrinsics["height"],
        )

        camera_spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K.reshape(-1),
            width=camera_setting["width"],
            height=camera_setting["height"],
            focal_length=float(focal_length * 10),
            focus_distance=600.0,
            clipping_range=(0.1, 1.0e5),
        )

    else:

        if "eyes" in camera_setting.keys():
            camera_rot = quat_from_matrix(
                create_rotation_matrix_from_view(
                    torch.as_tensor([camera_setting["eyes"]]),
                    torch.as_tensor([camera_setting["target"]])))[0]
            camera_translate = camera_setting["eyes"]
        else:
            if "quat" in camera_setting.keys():
                camera_rot = torch.as_tensor(camera_setting["quat"])
            else:
                euler = torch.as_tensor([camera_setting["euler"]
                                         ]) / 180.0 * torch.pi

                camera_rot = math_utils.quat_from_euler_xyz(
                    euler[:, 0], euler[:, 1], euler[:, 2])[0]

            camera_translate = camera_setting["translate"]
            # if camera_setting.get("converter") is not None:
            #     camera_rot = math_utils.convert_camera_frame_orientation_convention(
            #         camera_rot,
            #         origin=camera_setting["converter"],
            #         target="opengl")

        camera_spawn = sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            focus_distance=600.0,
            horizontal_aperture=20,
            clipping_range=(0.1, 20.0),
        )
    return camera_rot, camera_translate, camera_spawn


def configure_deformable_object(scene, object_name, object_config,
                                current_path):

    deformable_cfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + object_name,
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
            mass_props=sim_utils.MassPropertiesCfg(mass=200),
            deformable_props=DeformableBodyPropertiesCfg(
                simulation_hexahedral_resolution=object_config.get(
                    "simulation_hexahedral_resolution"),
                vertex_velocity_damping=object_config.get(
                    "vertex_velocity_damping"),
            ),
            semantic_tags=[("class", "deformable")],
        ),
        deform_cfg=object_config,
    )

    setattr(scene, object_name, deformable_cfg)


def configure_rigid_object(scene, object_name, object_config, current_path):
    if object_config.get("spawn", True):
        if object_config.get("transformation_txt") is None:
            if object_config.get("rot") is not None:
                rot = tuple(
                    obtain_target_quat_from_multi_angles(
                        object_config["rot"]["axis"],
                        object_config["rot"]["angles"]))
            elif object_config.get("quat") is not None:
                rot = object_config["quat"]
            pos = torch.as_tensor(object_config['pos'], dtype=torch.float16)

            scale = object_config["scale"]

        else:
            mat = get_transform_from_txt(object_config["transformation_txt"])
            pos = mat[:3, 3]
            rot = math_utils.quat_from_matrix(torch.as_tensor(
                mat[:3, :3])).numpy()
            scale = np.ones(3) * get_scale_from_transform(mat)
            object_config["scale"] = scale
            object_config["rot"]["quat"] = rot
            object_config["rot"]["pos"] = pos

        rigid_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + object_name,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"{current_path}/{object_config['path']}",
                scale=scale,
                # mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    disable_gravity=False,
                    kinematic_enabled=object_config.get(
                        "kinematic_enabled", False),
                ),
                semantic_tags=[("class", object_name)]),
            rigid_cfg=object_config,
        )

    else:
        scale = None

        rigid_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/" + object_config['path'],
            spawn=None,
        )

    setattr(scene, object_name, rigid_cfg)
    return object_config


def configure_articulation_object(scene, object_name, object_config,
                                  current_path):
    articulate_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}" + f"/{object_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=object_config["disable_gravity"]
                if "disable_gravity" in object_config.keys() else False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=object_config[
                    "enabled_self_collisions"] if "enabled_self_collisions"
                in object_config.keys() else False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0),
            scale=object_config["scale"],
            semantic_tags=[("class", object_name)]),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=object_config["pos"],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"])),
            joint_pos=object_config["joints"]),
        actuators={
            f"{object_name}":
            ImplicitActuatorCfg(
                joint_names_expr=[*object_config["joints"].keys()],
                effort_limit=200.0,
                velocity_limit=300.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
        },
        articulation_cfg=object_config)

    setattr(scene, object_name, articulate_cfg)


def configure_asset_object(scene, object_name, object_config, current_path):
    asset_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + object_name,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
        ),
    )
    setattr(scene, object_name, asset_cfg)


def configure_camera(scene,
                     camera_name,
                     camera_setting,
                     customize_camera_setting,
                     randomness=None,
                     camera_json=None):
    camera_rot, camera_translate, camera_spawn = load_camera_params(
        camera_name, camera_setting, camera_json)

    if camera_setting.get("prim_path", None) is not None:
        prim_path = "{ENV_REGEX_NS}/" + camera_setting["prim_path"]
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="ros")
    else:
        prim_path = "{ENV_REGEX_NS}/" + camera_name
        offset = CameraCfg.OffsetCfg(pos=camera_translate,
                                     rot=camera_rot,
                                     convention="opengl")

    camera_cfg = CameraCfg(
        prim_path=prim_path,
        offset=offset,
        update_period=camera_setting["update_period"],
        height=camera_setting["height"],
        width=camera_setting["width"],
        data_types=camera_setting["data_types"],
        spawn=camera_spawn,
        customize_camera_setting=randomness if randomness is not None else {},
        segmentation_point_cloud=customize_camera_setting[
            "segmentation_point_cloud"],
        colorize_point_cloud=customize_camera_setting["colorize_point_cloud"],
        num_downsampled_points=customize_camera_setting[
            "num_downsampled_points"],
        segmentation_point_cloud_id=customize_camera_setting[
            "segmentation_point_cloud_id"],
        debug_vis=True,
        colorize_instance_segmentation=False)

    setattr(scene, camera_name, camera_cfg)


def configure_raycaster_camera(scene,
                               camera_name,
                               camera_setting,
                               randomness=None,
                               camera_json=None):
    camera_rot, camera_translate, camera_spawn = load_camera_params(
        camera_name, camera_setting, camera_json)

    camera_cfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=RayCasterCfg.OffsetCfg(
            pos=camera_translate,
            rot=camera_rot,
        ),
        mesh_prim_paths=["/World/GroundPlane"],
        update_period=1 / 60,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        attach_yaw_only=True,
    )
    setattr(scene, camera_name, camera_cfg)


def configure_tiled_camera(scene,
                           camera_name,
                           camera_setting,
                           randomness=None,
                           camera_json=None):
    camera_rot, camera_translate, camera_spawn = load_camera_params(
        camera_name, camera_setting, camera_json)

    camera_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/" + "tile_" + camera_name,
        offset=TiledCameraCfg.OffsetCfg(pos=camera_translate,
                                        rot=camera_rot,
                                        convention="opengl"),
        # update_period=camera_setting["update_period"],
        height=camera_setting["height"],
        width=camera_setting["width"],
        data_types=camera_setting["data_types"],
        spawn=camera_spawn,
        customize_camera_setting=randomness if randomness is not None else {},
        colorize_instance_segmentation=False
        # debug_vis=True,
    )
    setattr(scene, camera_name, camera_cfg)


def configure_particle_object(scene, object_name, object_config, current_path):
    particle_cfg = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + {object_name},
        init_state=ParticleObjectCfg.InitialStateCfg(
            pos=object_config['pos'],
            rot=tuple(
                obtain_target_quat_from_multi_angles(
                    object_config["rot"]["axis"],
                    object_config["rot"]["angles"]))),
        spawn=UsdFileCfg(
            usd_path=f"{current_path}/{object_config['path']}",
            scale=object_config['scale'],
            particle_props=ParticleBodyPropertiesCfg(),
        ),
        deform_cfg=object_config,
    )
    setattr(scene, object_name, particle_cfg)


def set_from_config(scene, env_cfg, current_path):
    # Configure Deformable Objects

    deformable_objects = env_cfg['params'].get('DeformableObject', {})
    if deformable_objects is not None:
        for object_name, object_config in deformable_objects.items():
            configure_deformable_object(scene, object_name, object_config,
                                        current_path)

    # Configure Asset Objects
    asset_objects = env_cfg['params'].get('AssstObject', {})
    if asset_objects is not None:
        for object_name, object_config in asset_objects.items():
            configure_asset_object(scene, object_name, object_config,
                                   current_path)

    # Configure Articulation Objects
    articulation_objects = env_cfg['params'].get('ArticulationObject', {})
    if articulation_objects is not None:
        for object_name, object_config in articulation_objects.items():
            configure_articulation_object(scene, object_name, object_config,
                                          current_path)

    # Configure Rigid Objects
    rigid_objects = env_cfg['params'].get('RigidObject', {})

    if rigid_objects is not None:
        for object_name, object_config in rigid_objects.items():

            configure_rigid_object(scene, object_name, object_config,
                                   current_path)

    # Configure Cameras
    if "Camera" in env_cfg['params'].keys():
        if env_cfg['params']['Camera'].get("initial", False):

            for camera_name, camera_setting in env_cfg['params']['Camera'][
                    "cameras"].items():

                camera_type = camera_setting["camera_type"]

                if camera_type == "camera":
                    configure_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        customize_camera_setting=env_cfg['params']['Camera']
                        ["pointcloud_setting"],
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])
                elif camera_type == "tiled_camera":
                    configure_tiled_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])
                elif camera_type == "ray_caster_camera":
                    configure_raycaster_camera(
                        scene,
                        camera_name,
                        camera_setting,
                        randomness=env_cfg['params']['Camera']
                        ["random_pose_setting"]
                        if env_cfg['params']['Camera'].get(
                            "random_pose", False) else None,
                        camera_json=env_cfg['params']['Camera']["camera_json"])
        # Configure Particle Objects

    particle_objects = env_cfg['params'].get('ParticlesObject', {})
    if particle_objects is not None:
        for object_name, object_config in particle_objects.items():
            configure_particle_object(scene, object_name, object_config,
                                      current_path)
