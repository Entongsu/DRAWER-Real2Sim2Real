from robomimic.utils.test_utils import test_eval_agent_from_checkpoint
import robomimic.utils.file_utils as FileUtils

import pdb

import argparse
import json

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *
import cv2
from robosuite.utils.camera_utils import *
from robosuite.utils.visualization_utils import vis_pc
import pdb
from robosuite.utils.transform_utils import *
from robosuite.controllers.curobo_planner import *
import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml
import cv2
import imageio

# from robosuite.utils.mesh_utils import *
from robosuite.data_collection.speed_robotpcd import *

from robomimic.utils.error_utils import *

image_buffer = None


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


import pickle


def save_file(file_name, data):
    with open(file_name, 'wb') as file:
        # Use pickle.dump() to save the object to the file
        pickle.dump(data, file)


def read_calibration_file(filename):

    with open(filename) as file:
        calib_file = json.load(file)

    calib_dict = {}
    for calib in calib_file:
        sn = calib["camera_serial_number"]
        calib_dict[sn] = {"intrinsic": {}, "extrinsic": {}}
        calib_dict[sn]["intrinsic"] = calib["intrinsics"]
        calib_dict[sn]["extrinsic"]["pos"] = calib["camera_base_pos"]
        calib_dict[sn]["extrinsic"]["ori"] = calib["camera_base_ori"]
    return calib_dict


def read_data(file_name, data_type):
    with open(file_name + f"/{data_type}.pkl", 'rb') as file:
        # Use pickle.load() to read the object from the file
        loaded_data = pickle.load(file)
    return loaded_data


def read_camera_data(file_name):
    with open(file_name, 'rb') as file:
        # Use pickle.load() to read the object from the file
        loaded_data = pickle.load(file)
    return loaded_data


def process_image(image, image_name="frontview_image"):

    image = cv2.flip(obs0[image_name], 0)

    # cv2.imshow('Real-time video', image)
    # cv2.waitKey(1)

    # # Press 'q' on the keyboard to exit the loop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    transformed_image = np.transpose(image, (2, 0, 1))
    return transformed_image / 255


if __name__ == "__main__":
    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=["train", "test", "replay", "train_obs"])
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--use_pc",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--num_pc",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--use_rgb",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--show_error",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    device = "cuda:0"
    ckpt_path = "/media/lme/data2/weird/robomimic/bc_trained_models/test/20240508112930/models/model_epoch_200.pth"
    file_name = '/media/lme/data2/weird/training/poly/demo_lift_pcd_speed1024.h5py'
    if args.use_rgb:
        camera_heights = 80
        camera_widths = 160
    else:
        camera_heights = 320
        camera_widths = 640

    # init error buffer
    if args.show_error:
        error_function = Error(
            error_path="/media/lme/data2/weird/training/error/")

    cube_mesh = trimesh.load("/media/lme/data2/weird/new_cube.stl")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path,
                                                         device=device,
                                                         verbose=True)

    renderer = "mujoco"
    # # Load the desired controller
    controller_type = "OSC_POSE"

    options = {
        'env_name': 'Lift',
        'robots': 'Panda',
    }
    options["controller_configs"] = load_controller_config(
        default_controller=controller_type)

    camera_names = ["frontview", "leftview"]
    use_camera_obs = True
    env = suite.make(
        **options,
        has_renderer=False
        if renderer != "mujoco" else True,  # no on-screen renderer
        # has_renderer=False,
        has_offscreen_renderer=True,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=use_camera_obs,  # no camera observations
        control_freq=20,
        renderer=renderer,
        camera_names=camera_names,
        camera_depths=True,
        camera_segmentations=
        None,  #['class','class'],  #['instance', 'class', 'element'],
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        render_camera="leftview",
        hard_reset=False,
        initialization_noise={
            "magnitude": 0.05,
            "type": "gaussian"
        })

    # env.viewer.set_camera(camera_id=0)

    calib_dict = read_calibration_file(
        "/media/lme/data2/weird/robosuite/robosuite/caliberation/default.json")
    obs = env.reset(calib_dict=calib_dict,
                    camera_names=["frontview", "leftview"])

    low, high = env.action_spec
    action = np.random.uniform(low, high) * 0
    obs, reward, done, _ = env.step(action)

    cameras_dict = read_camera_data(
        "/media/lme/data2/weird/data_collection//camera.pkl")

    writer = imageio.get_writer("test.mp4")

    buffers = []
    indices = []

    if args.use_pc:
        robot_arm_names = [
            'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
            'link7', "eef", "leftfinger", "rightfinger"
        ]
        robot_arm_meshes = get_robot_arm_mesh(robot_arm_names)

    if args.mode == "train" or args.mode == "replay" or args.mode == "train_obs":
        file = h5py.File(file_name, 'r')

        dataset = file["data"]
        num_traj = len(dataset.keys())

    else:
        num_traj = 100

    for _ in range(args.start, num_traj):
        num = np.random.randint(low=100, high=200)

        if args.mode == "train" or args.mode == "replay" or args.mode == "train_obs":

            demonstration = dataset[f"demo_{num}"]
            actions = demonstration["actions"]
            dones = demonstration["dones"]
            next_obs = demonstration["next_obs"]
            obs = demonstration["obs"]
            rewards = demonstration["rewards"]

            reset_joint = obs["robot0_joint_pos"][0]

            env.robots[0].set_robot_joint_positions(reset_joint)

            horizon = len(actions)
           

            env.sim.data.set_joint_qpos(env.cube.joints[0],np.concatenate([ np.array(obs["cube_pos"][0]),np.array(obs["cube_quat"][0])[[3, 0, 1, 2]]]))

            # if args.mode == "train" and not args.show_error:

            #     horizon = args.horizon

        else:
            horizon = args.horizon

        obs0, reward, done, _ = env.step(action)

        if args.use_pc:
            downsampled_points = batch_preprocess_pc(
                [obs0["leftview_depth"]], [obs0["frontview_depth"]],
                [obs0["robot_arm_pos"]], [obs0["robot_arm_quat"]],
                [obs0["cube_pos"]], [obs0["cube_quat"]], [obs0["cube_size"]],
                cube_mesh,
                robot_arm_meshes,
                robot_arm_names,
                cameras_dict,
                downsample_robot_points=300,
                downsample_cube_points=1000,
                downsample_real_points=args.num_pc)
            obs0["pcd"] = np.transpose(downsampled_points[0], (1, 0))

        # for index, action in enumerate(actions):
        for index in range(horizon):
            if args.mode == "train":
                if env.use_camera_obs and args.use_rgb:
                    processed_image = process_image(obs0["frontview_image"])
                    obs0["frontview_image"] = processed_image

                if args.use_pc:
                    downsampled_points = batch_preprocess_pc(
                        [obs0["leftview_depth"]], [obs0["frontview_depth"]],
                        [obs0["robot_arm_pos"]], [obs0["robot_arm_quat"]],
                        [obs0["cube_pos"]], [obs0["cube_quat"]],
                        [obs0["cube_size"]],
                        cube_mesh,
                        robot_arm_meshes,
                        robot_arm_names,
                        cameras_dict,
                        downsample_robot_points=300,
                        downsample_cube_points=1000,
                        downsample_real_points=args.num_pc)
                    obs0["pcd"] = np.transpose(downsampled_points[0], (1, 0))

                # writer.append_data(cv2.flip(obs0["frontview_image"], 0))

                ac = policy(ob=obs0)

                if args.show_error:
                    error_function.get_compounding_error(
                        obs["robot0_eef_pos"][index], obs0["robot0_eef_pos"])

            elif args.mode == "replay":
                if args.use_pc:
                    downsampled_points = batch_preprocess_pc(
                        [obs0["leftview_depth"]], [obs0["frontview_depth"]],
                        [obs0["robot_arm_pos"]], [obs0["robot_arm_quat"]],
                        [obs0["cube_pos"]], [obs0["cube_quat"]],
                        [obs0["cube_size"]],
                        cube_mesh,
                        robot_arm_meshes,
                        robot_arm_names,
                        cameras_dict,
                        downsample_robot_points=300,
                        downsample_cube_points=1000,
                        downsample_real_points=args.num_pc)
                    obs0["pcd"] = np.transpose(downsampled_points[0], (1, 0))

                pre_ac = policy(ob=obs0)
                # print(np.max(pre_ac),np.max(actions[index]))

                ac = actions[index]

            elif args.mode == "test":

                if env.use_camera_obs and args.use_rgb:
                    processed_image = process_image(obs0["frontview_image"])
                    obs0["frontview_image"] = processed_image

                ac = policy(ob=obs0)
                ac[-1] = np.sign(ac[-1])

            elif args.mode == "train_obs":

                observaion = {
                    # "cube_pos": obs["cube_pos"][index],
                    # "cube_quat": obs["cube_quat"][index],
                    "robot0_joint_pos": obs["robot0_joint_pos"][index],
                    "robot0_eef_quat": obs["robot0_eef_quat"][index],
                    "robot0_eef_pos": obs["robot0_eef_pos"][index],
                    "target_placement_pos": obs["target_placement_pos"][index],
                    "target_placement_quat":
                    obs["target_placement_quat"][index],
                    "robot0_joint_pos": obs["robot0_joint_pos"][index],
                    # "frontview_image": obs["frontview_image"][index],
                    "pcd": np.transpose(obs["pcd"][index], (1, 0)),
                    "robot0_gripper_qpos": obs["robot0_gripper_qpos"][index]
                }

                if args.use_rgb:
                    observaion["frontview_image"] = obs["frontview_image"][
                        index]

                    observaion[
                        "frontview_image"] = transformed_image = np.transpose(
                            observaion["frontview_image"], (2, 0, 1)) / 255

                ac = policy(ob=observaion)

            ac[-1] = np.sign(ac[-1])

            obs0, reward, done, _ = env.step(ac)
            if args.use_pc:
                downsampled_points = batch_preprocess_pc(
                    [obs0["leftview_depth"]], [obs0["frontview_depth"]],
                    [obs0["robot_arm_pos"]], [obs0["robot_arm_quat"]],
                    [obs0["cube_pos"]], [obs0["cube_quat"]],
                    [obs0["cube_size"]],
                    cube_mesh,
                    robot_arm_meshes,
                    robot_arm_names,
                    cameras_dict,
                    downsample_robot_points=300,
                    downsample_cube_points=1000,
                    downsample_real_points=args.num_pc)
                obs0["pcd"] = np.transpose(downsampled_points[0], (1, 0))

            env.render()

        if args.show_error:
            error_function.visualize()

        obs = env.reset(calib_dict=calib_dict,
                        camera_names=["frontview", "leftview"])

        open_gripper(env)
        env.reset()

        for i in range(10):
            obs, reward, done, _ = env.step(np.zeros(7))

    print("Done.")
