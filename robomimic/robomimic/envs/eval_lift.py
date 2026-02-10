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

from robosuite.utils.mesh_utils import *

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


def get_robot_arm_mesh_from_robosuite():

    robot_arm_names = [
        'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
        "eef", "leftfinger", "rightfinger"
    ]

    robot_arm_meshes = {}

    for name in robot_arm_names:
        mesh_path =  f"/media/lme/data2/weird/robosuite/robosuite/models/assets/robots/panda/obj_meshes/{name}_vis/"
        if name in ["eef", "leftfinger", "rightfinger"]:
            mesh_path = f"/media/lme/data2/weird/data_pcd/upsample_{name}_vis/"

        mesh_files = os.listdir(
           mesh_path
        )
        meshes = []
        for mesh_file in mesh_files:
            if mesh_file.endswith(".mtl"):
                continue
            mesh = trimesh.load(
                f"/media/lme/data2/weird/robosuite/robosuite/models/assets/robots/panda/obj_meshes/{name}_vis/"
                + mesh_file)
            meshes.append(mesh)

        combined_mesh = trimesh.util.concatenate(meshes)
        # combined_mesh.export(f"/media/lme/data2/weird/trash/ori_{name}.stl")
        robot_arm_meshes[name] = combined_mesh

    return robot_arm_meshes




def transform_robot_arm2(xposes, xquates, robot_arm_meshes, downsample=None):

    robot_arm_names = [
        'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
        "eef", "leftfinger", "rightfinger"
    ]

    all_pc = []
    xposes = xposes[-len(robot_arm_names):]
    xquates = xquates[-len(robot_arm_names):]
   
    for i in range(len(xposes)):

        xpos = xposes[i]

        xquat = xquates[i][[3, 0, 1, 2]]  # xyzw->wxyz
        # if "eef" in key:
        #     xquat = [1, 0, 0, 0]
        transformation_matrix = trimesh.transformations.quaternion_matrix(
            xquat)

        transformation_matrix[:3, 3] = xpos

        mesh = robot_arm_meshes[robot_arm_names[i]].copy()
        new_vertices = pc_transformation(mesh.vertices, transformation_matrix)

        if downsample is not None:
            choice = np.random.choice(len(new_vertices),
                                      downsample,
                                      replace=False)
            new_vertices = new_vertices[choice]

        all_pc.append(new_vertices)

    pointcloud = np.concatenate(all_pc, axis=0)
    return pointcloud


def preprocess_pc(obs,
                  downsample_robot=300,
                  downsample_object=1000,
                  num_points=6000,
                  cube_mesh=None):

    roboot_pointcloud = transform_robot_arm2(obs["robot_arm_pos"],
                                            obs["robot_arm_quat"],
                                            robot_arm_meshes,
                                            downsample=downsample_robot)

    cube_mesh = transform_mesh_to_pcd(obs["cube_pos"],
                                      obs["cube_quat"],
                                      obs["cube_size"],
                                      cube_mesh,
                                      downsample=1000)

    pc = depth_to_pcd(cameras_dict, ["frontview", "leftview"], [
        obs["frontview_depth"],
        obs["leftview_depth"],
    ], [-0.1, -0.4, 0.005], [1.0, 0.4, 1.4],
                      num_points=num_points)

    # pc = depth_to_pcd(cameras_dict, ["frontview", "leftview"], [
    #     obs["frontview_depth"],
    #     obs["leftview_depth"],
    # ], [-1, -1, -1], [1.0, 1.4, 1.4],
    #                   num_points=num_points)

    all_pc = np.concatenate([pc, roboot_pointcloud, cube_mesh], axis=0)
    downsampled_points = downsample_internal(all_pc,
                                             feat=None,
                                             num_points=num_points)

    np.save("test.npy", all_pc)
    # visualize_pcd(downsampled_points, video_writer=None)

    return np.transpose(downsampled_points, (1, 0))


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
        default=150,
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
    ckpt_path = "/media/lme/data2/weird/robomimic/bc_trained_models/test/20240505121734/models/model_epoch_200.pth"
    file_name = '/media/lme/data2/weird/training/h5py/demo_lift_pcd_speed1024.h5py'
    if args.use_rgb:
        camera_heights = 80
        camera_widths = 160
    else:
        camera_heights = 320
        camera_widths = 640
        
    # init error buffer
    if args.show_error:
        error_function = Error(error_path="/media/lme/data2/weird/training/error/")

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

    buffers = []
    indices = []

    if args.use_pc:
        robot_arm_meshes = get_robot_arm_mesh_from_robosuite()

    if args.mode == "train" or args.mode == "replay" or args.mode == "train_obs":
        file = h5py.File(file_name, 'r')

        dataset = file["data"]
        num_traj = len(dataset.keys())

    else:
        num_traj = 100

    for num in range(num_traj):

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
            
            if args.mode == "train" and not args.show_error:
                
                horizon = args.horizon
           
                
        else:
            horizon = args.horizon

        obs0, reward, done, _ = env.step(action)

        if args.use_pc:
            obs0["pcd"] = preprocess_pc(obs0, num_points=args.num_pc, cube_mesh=cube_mesh)

        # for index, action in enumerate(actions):
        for index in range(horizon):
            if args.mode == "train":
                if env.use_camera_obs and args.use_rgb:
                    processed_image = process_image(obs0["frontview_image"])
                    obs0["frontview_image"] = processed_image
                
                if args.use_pc:
                    obs0["pcd"] = preprocess_pc(obs0,
                                                num_points=args.num_pc,
                                                cube_mesh=cube_mesh)
                    
            
                
                #np.save("test.npy", np.transpose(np.concatenate([obs0["pcd"],np.transpose(obs["pcd"][index], (1, 0))],axis=1),(1,0)))
                # # import pdb
                # pdb.set_trace()
                
                # print(aa)
                # print(index)
                # np.save("test12.npy", np.transpose(obs0["pcd"],(1,0)))
                # import pdb
                # pdb.set_trace()
                # print(abs(obs["cube_pos"][index][2]-obs["cube_pos"][0][2]))
                # if abs(obs["cube_pos"][index][2]-obs["cube_pos"][0][2]) >0.02:
                #     pdb.set_trace()
               
                ac = policy(ob=obs0)
                
                if args.show_error:
                    error_function.get_compounding_error(obs["robot0_eef_pos"][index],obs0["robot0_eef_pos"])

            elif args.mode == "replay":
                if args.use_pc:
                    obs0["pcd"] = preprocess_pc(obs0,
                                                num_points=args.num_pc,
                                                cube_mesh=cube_mesh)
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
                    # "frontview_image": obs["frontview_image"][index],
                    "pcd": np.transpose(obs["pcd"][index], (1, 0)),
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
                obs0["pcd"] = preprocess_pc(obs0,
                                            num_points=args.num_pc,
                                            cube_mesh=cube_mesh)

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
