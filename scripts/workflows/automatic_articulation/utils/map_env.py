import torch
from scripts.workflows.automatic_articulation.utils.load_obj_utils import save_robot_mesh
import trimesh
import sys
try:
    sys.path.append("/media/lme/data4/weird/Grounded-SAM-2")
    from sam_tool import ObjectSegmenter
except:
    pass
env_map = {
    'init_open': 'env_cabinet',
    'init_close': 'env_cabinetclose',
    'init_grasp': 'env_grasp'
}
# Map collect flags to their corresponding environments
collect_map = {
    'collect_cabinet': 'env_cabinet',
    'collect_close': 'env_cabinetclose',
    'collect_grasp': 'env_grasp',
    'collect_placement': 'env_placement'
}

reset_map = {
    'reset_cabinet': 'env_cabinet',
    'reset_close': 'env_cabinetclose',
    'reset_grasp': 'env_grasp',
    'reset_placement': 'env_placement'
}

# cache_map = {
#     'grasp_success': ('grasp_collector_interface', [
#         'pick_obs_buffer', 'pick_actions_buffer', 'pick_rewards_buffer',
#         'pick_does_buffer'
#     ]),
#     'close_success': ('close_collector_interface', [
#         'close_obs_buffer', 'close_actions_buffer', 'close_rewards_buffer',
#         'close_does_buffer'
#     ]),
#     'cabinet_success': ('cabinet_collector_interface', [
#         'cabinet_obs_buffer', 'cabinet_actions_buffer',
#         'cabinet_rewards_buffer', 'cabinet_does_buffer'
#     ])
# }

# Map collection flags to their corresponding buffers
step_buffer_map = {
    'collect_grasp': [
        'pick_obs_buffer', 'pick_actions_buffer', 'pick_rewards_buffer',
        'pick_does_buffer'
    ],
    'collect_cabinet': [
        'cabinet_obs_buffer', 'cabinet_actions_buffer',
        'cabinet_rewards_buffer', 'cabinet_does_buffer'
    ],
    'collect_placement': [
        'pick_obs_buffer', 'pick_actions_buffer', 'pick_rewards_buffer',
        'pick_does_buffer'
    ],
    'collect_close': [
        'close_obs_buffer', 'close_actions_buffer', 'close_rewards_buffer',
        'close_does_buffer'
    ]
}

# Map reset flags to buffer attributes
reset_buffer_map = {
    'reset_cabinet': [
        'cabinet_obs_buffer', 'cabinet_actions_buffer',
        'cabinet_rewards_buffer', 'cabinet_does_buffer'
    ],
    'reset_grasp': [
        'pick_obs_buffer', 'pick_actions_buffer', 'pick_rewards_buffer',
        'pick_does_buffer'
    ],
    'reset_placement': [
        'place_obs_buffer', 'place_actions_buffer', 'place_rewards_buffer',
        'place_does_buffer'
    ],
    'reset_close': [
        'close_obs_buffer', 'close_actions_buffer', 'close_rewards_buffer',
        'close_does_buffer'
    ]
}


def reset_data_buffer(object,
                      reset_cabinet=False,
                      reset_grasp=False,
                      reset_placement=False,
                      reset_close=False,
                      presave_buffer=None):

    # Dynamically reset the buffers based on flags
    for reset_flag, buffers in reset_buffer_map.items():
        if locals()[reset_flag]:  # Check the corresponding reset flag
            for index, buffer in enumerate(buffers):
                if presave_buffer is not None:

                    setattr(object, buffer, presave_buffer[index])
                else:
                    setattr(object, buffer, [])


robot_body_names = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
    'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand',
    'panda_leftfinger', 'panda_rightfinger'
]


def load_robot_mesh(log_dir):
    mesh_dict = {}
    for body_id, body_name in enumerate(robot_body_names):
        mesh = trimesh.load(f"logs/mesh/{body_name}.obj")

        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        mesh_dict[body_name] = vertices
    return mesh_dict


# def update_success_flags(obj):
#     # Mapping init flags to corresponding success flags
#     success_map = {
#         'init_open': 'cabinet_success',
#         'init_close': 'close_success',
#         'init_grasp': 'grasp_success',
#         'init_placement': 'placement_success'
#     }

#     # Update success flags based on init flags
#     for init_flag, success_flag in success_map.items():
#         setattr(obj, success_flag, getattr(obj, init_flag, False))


def load_config(obj):
    # Dynamically set attributes based on the configuration
    config_map = {
        "seg_target_name":
        obj.env_config["params"]["Task"]["seg_target_name"],
        "use_bounding_box":
        obj.env_config["params"]["Task"]["use_bounding_box"],
        "bbox_range":
        obj.env_config["params"]["Task"]["bbox_range"],
        "segment_handle":
        obj.env_config["params"]["Task"]["segment_handle"],
        "segment_handle_camera_id":
        obj.env_config["params"]["Task"]["segment_handle_camera_id"],
        "use_fps":
        obj.env_config["params"]["Task"]["use_fps"],
        "aug_robot_mesh":
        obj.env_config["params"]["Task"]["aug_robot_mesh"]
    }

    for attr, value in config_map.items():
        setattr(obj, attr, value)

    # Additional settings
    if obj.use_bounding_box:
        setattr(obj, "init_bbox", None)

    if obj.aug_robot_mesh:

        save_robot_mesh(obj.args_cli.log_dir, obj.env)

        setattr(obj, "robot_mesh", load_robot_mesh(obj.args_cli.log_dir))

    if obj.segment_handle:
        setattr(
            obj, "sam_tool",
            ObjectSegmenter(
                obj.device,
                box_threshold=0.25,
                text_threshold=0.25,
            ))
        setattr(obj, "handle_mask", None)
    placement_region = obj.env_config["params"]["Task"]["placement"].get(
        "placement_region", None)
    setattr(obj, "placement_region", placement_region)


def init_setting(obj):
    # Robot settings
    setattr(obj, "target_handle_name",
            obj.kitchen.cfg.articulation_cfg["target_drawer"])
    setattr(
        obj, "robot_offset",
        torch.as_tensor(obj.kitchen.cfg.articulation_cfg["robot_random_range"][
            obj.target_handle_name]["offset"]).to(obj.device))
    setattr(
        obj, "robot_pose_random_range",
        obj.kitchen.cfg.articulation_cfg["robot_random_range"][
            obj.target_handle_name]["pose_range"])

    if "pose_range" in obj.kitchen.cfg.articulation_cfg.keys():
        setattr(obj, "kitchen_pose_range",
                obj.kitchen.cfg.articulation_cfg["pose_range"])
    else:
        setattr(obj, "kitchen_pose_range", None)

    # Target object setting
    setattr(obj, "grasp_object",
            obj.kitchen.cfg.articulation_cfg["target_object"])

    handle_id, handle_name = obj.kitchen.find_bodies(obj.target_handle_name)
    setattr(obj, "handle_id", handle_id)
    setattr(obj, "handle_name", handle_name)

    joint_ids, joint_names = obj.kitchen.find_joints(
        obj.kitchen.cfg.articulation_cfg["robot_random_range"][
            obj.target_handle_name]["joint_name"])
    setattr(obj, "joint_ids", joint_ids)
    setattr(obj, "joint_names", joint_names)

    setattr(
        obj, "ee_random_range",
        obj.kitchen.cfg.articulation_cfg["robot_random_range"][
            obj.target_handle_name]["ee_pos_random_range"])
    setattr(
        obj, "joint_random_range",
        obj.kitchen.cfg.articulation_cfg["robot_random_range"][
            obj.target_handle_name]["joint_random_range"])
    setattr(obj, "target_joint_type",
            obj.kitchen.cfg.articulation_cfg["target_joint_type"])

    # Camera settings
    setattr(obj, "randomize_camera_pose",
            obj.env_config["params"]["Task"]["randomize_camera_pose"])
    setattr(obj, "randomize_camera_pose_range",
            obj.env_config["params"]["Task"]["randomize_camera_pose_range"])
    setattr(obj, "curobo_planner_length",
            obj.env_config["params"]["Task"]["curobo_planner_length"])
    setattr(obj, "success_pick_threhold",
            obj.env_config["params"]["Task"]["success_pick_threhold"])


def placement_region(placement_pose):

    # Convert min and max to torch tensors
    min_vals = torch.tensor(placement_pose['min'])
    max_vals = torch.tensor(placement_pose['max'])

    # Generate a random 3D tensor within the range
    random_pose = min_vals + (max_vals - min_vals) * torch.rand(3)
    return random_pose
