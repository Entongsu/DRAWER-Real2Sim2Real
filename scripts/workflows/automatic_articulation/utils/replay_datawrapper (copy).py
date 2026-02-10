from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from tools.visualization_utils import vis_pc, visualize_pcd
import imageio
from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose
import cv2
import sys
from tools.curobo_planner import MotionPlanner, IKPlanner
try:
    sys.path.append("/home/ensu/Documents/weird/Grounded-SAM-2")
    from sam_tool import ObjectSegmenter
except:
    pass
import json
import pytorch3d.ops as torch3d_ops
# from torch_cluster import fps
# from dgl.geometry import farthest_point_sampler
import pathlib
import os
import copy
import isaacsim.core.utils.prims as prim_utils
from curobo.util.usd_helper import UsdHelper
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import trimesh
from omegaconf import OmegaConf
from tools.visualization_utils import *


def get_prim_world_pose(cache: UsdGeom.XformCache,
                        prim: Usd.Prim,
                        inverse: bool = False):
    world_transform: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    # get scale:
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    scale = list(scale)
    t_mat = world_transform.RemoveScaleShear()
    if inverse:
        t_mat = t_mat.GetInverse()

    # mat = np.zeros((4,4))
    # mat[:,:] = t_mat
    translation: Gf.Vec3d = t_mat.ExtractTranslation()
    rotation: Gf.Rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    orientation = [q.GetReal()] + list(q.GetImaginary())
    x, y, z = translation[0], translation[1], translation[2]

    return [x, y, z] + orientation + scale


def save_robot_mesh(log_dir, env):
    os.makedirs(log_dir + "/mesh", exist_ok=True)
    usd_help = UsdHelper()
    usd_help.load_stage(env.scene.stage)

    all_items = usd_help.stage.Traverse()
    visual_prim = [
        x for x in all_items if "Robot" in x.GetPath().pathString
        and "visual" in x.GetPath().pathString
    ]
    robot_link_raw_trans = {}
    for prim in visual_prim:

        prim_path = prim.GetPath().pathString

        visual_mesh_prim = prim_utils.get_prim_at_path(
            prim_path + "/" + prim_path.split("/")[-2])
        points = list(visual_mesh_prim.GetAttribute("points").Get())
        points = [np.ravel(x) for x in points]
        faces = list(visual_mesh_prim.GetAttribute("faceVertexIndices").Get())
        face_count = list(
            visual_mesh_prim.GetAttribute("faceVertexCounts").Get())
        faces = np.array(faces).reshape(len(face_count), 3)
        points = np.array(points)
        trans_info = get_prim_world_pose(usd_help._xform_cache,
                                         visual_mesh_prim)

        link_mesh = trimesh.Trimesh(points * trans_info[-1], faces)
        link_mesh.export(log_dir + "/mesh/" + prim_path.split("/")[-2] +
                         ".obj")
        robot_link_raw_trans[prim_path.split("/")[-2]] = (trans_info)
    np.save(log_dir + "/mesh/robot_link_raw_pose.npy", robot_link_raw_trans)


class ReplayDatawrapper:

    def __init__(self,
                 env,
                 init_grasp=False,
                 init_open=False,
                 init_placement=False,
                 init_close=False,
                 collect_data=False,
                 args_cli=None,
                 filter_keys=None,
                 replay=False,
                 use_relative_pose=False,
                 policy_path=None,
                 num_samples_points=2048,
                 eval=False,
                 replay_normalized_actions=False,
                 add_action_noise=False,
                 action_noise=[0, 0],
                 eval_3d_diffusion=False,
                 use_joint_pos=False):
        self.add_action_noise = add_action_noise
        self.env = env
        self.init_open = init_open
        self.init_grasp = init_grasp
        self.init_placement = init_placement
        self.init_close = init_close
        self.replay = replay
        self.device = self.env.device
        self.use_relative_pose = use_relative_pose
        self.args_cli = args_cli
        self.collect_data = collect_data
        self.num_samples_points = num_samples_points

        self.eval = eval

        self.replay_normalized_actions = replay_normalized_actions
        self.action_noise = action_noise
        self.eval_3d_diffusion = eval_3d_diffusion

        self.use_joint_pos = use_joint_pos
        if self.use_joint_pos:
            self.init_planner()

        self.filter_keys = filter_keys
        self.init_setting(args_cli, init_grasp, init_open, init_placement,
                          init_close, filter_keys, collect_data)

        if policy_path is not None:
            # if not eval_bc:
            self.load_policy(policy_path)
            self.policy_path = policy_path

    def init_planner(self):
        self.planner = IKPlanner(env=None, )

    def get_segmentation_label(self):
        self.segmentation_label = []
        camera_count = 0
        for _, key in enumerate(self.env.scene.keys()):
            if "camera" in key:

                idToLabels = self.env.scene[key].data.info[
                    "instance_segmentation_fast"]["idToLabels"]

                for key in idToLabels.keys():
                    for target_seg_key in self.seg_target_name:
                        if target_seg_key in idToLabels[key]:
                            self.segmentation_label.append(int(key))

                camera_count += 1

        self.segmentation_label = torch.unique(
            torch.tensor(self.segmentation_label)).to(self.device)

    def get_checkpoint_path(self, output_dir, tag='latest'):
        if tag == 'latest':
            return pathlib.Path(output_dir).joinpath('checkpoints',
                                                     f'{tag}.ckpt')
        elif tag == 'best':
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(
                    ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self,
                     payload,
                     exclude_keys=None,
                     include_keys=None,
                     **kwargs):
        import dill
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_policy(self, ckpt_path):
        if not self.eval_3d_diffusion:
            import sys
            sys.path.append("../robomimic")
            import robomimic.utils.file_utils as FileUtils
            self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(
                ckpt_path=ckpt_path, device=self.device, verbose=True)

            self.policy.start_episode()
        else:
            import dill
            import hydra
            from omegaconf import OmegaConf
            import pdb
            config = OmegaConf.load(
                "../3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml"
            )
            # config = OmegaConf.load(
            #     "../3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/pointnet_policy.yaml"
            # )

            env_config = OmegaConf.load(
                "../3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/kitchen_grasp.yaml"
            )
            cfg = OmegaConf.merge(config, env_config)

            self.model = hydra.utils.instantiate(cfg.policy)
            self.ema_model = None
            if cfg.training.use_ema:
                try:
                    self.ema_model = copy.deepcopy(self.model)
                except:  # minkowski engine could not be copied. recreate it
                    self.ema_model = hydra.utils.instantiate(cfg.policy)

            path = pathlib.Path(ckpt_path)
            payload = torch.load(path.open('rb'),
                                 pickle_module=dill,
                                 map_location='cpu')
            self.load_payload(payload,
                              exclude_keys=["optimizer"],
                              include_keys=None)

            self.policy = self.model
            if cfg.training.use_ema:
                self.policy = self.ema_model
            self.policy.eval()
            self.policy.cuda()

    def init_setting(self, args_cli, init_grasp, init_open, init_placement,
                     init_close, filter_keys, collect_data):
        self.robot = self.env.scene["robot"]
        self.kitchen = self.env.scene["kitchen"]
        self.target_handle_name = self.env.scene[
            "kitchen"].cfg.articulation_cfg["target_drawer"]
        self.grasp_object = self.env.scene[
            self.env.scene["kitchen"].cfg.articulation_cfg["target_object"]]
        self.random_camera = self.env.scene["kitchen"].cfg.articulation_cfg[
            "random_camera"]
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        self.demo_index = 0
        self.handle_id, handle_name = self.kitchen.find_bodies(
            self.target_handle_name)
        self.joint_ids, joint_names = self.kitchen.find_joints(
            self.kitchen.cfg.articulation_cfg["robot_random_range"][
                self.target_handle_name]["joint_name"])
        self.load_config()

        self.collector_interface = MultiDatawrapper(
            args_cli,
            init_grasp,
            init_open,
            init_placement,
            init_close,
            self.env_config,
            filter_keys,
            collect_data,
            replay=self.replay,
            eval=self.eval,
            use_fps=self.use_fps,
            replay_normalized_actions=self.replay_normalized_actions,
            add_action_noise=self.add_action_noise,
            action_noise=self.action_noise,
            aug_robot_mesh=self.aug_robot_mesh,
            use_joint_pos=self.use_joint_pos)
        self.collector_interface.load_h5py()

        if self.init_grasp:
            self.env_cfg = self.collector_interface.grasp_raw_data[
                "data"].attrs["env_setting"]
        elif self.init_close:
            self.env_cfg = self.collector_interface.close_raw_data[
                "data"].attrs["env_setting"]
        elif self.init_open:
            self.env_cfg = self.collector_interface.cabinet_raw_data[
                "data"].attrs["env_setting"]
        self.env_cfg = json.loads(self.env_cfg)

        if collect_data:
            self.reset_data_buffer(reset_cabinet=True,
                                   reset_grasp=True,
                                   reset_placement=True,
                                   reset_close=True)

        self.target_drawer = self.env_cfg["params"]["ArticulationObject"][
            "kitchen"]["target_drawer"]
        self.robot_random_pose_range = self.env_cfg["params"][
            "ArticulationObject"]["kitchen"]["robot_random_range"][
                self.target_drawer]
        self.target_object = self.env_cfg["params"]["ArticulationObject"][
            "kitchen"]["target_object"]

        self.target_object_random_pose_range = self.env_cfg["params"][
            "RigidObject"][self.target_object]['pose_range']
        self.target_joint_type = self.kitchen.cfg.articulation_cfg[
            "target_joint_type"]

    def load_config(self):

        self.env_config = OmegaConf.load(self.args_cli.config_file)
        self.seg_target_name = self.env_config["params"]["Task"][
            "seg_target_name"]
        self.use_bounding_box = self.env_config["params"]["Task"][
            "use_bounding_box"]
        self.bbox_range = self.env_config["params"]["Task"]["bbox_range"]
        if self.use_bounding_box:
            self.init_bbox = None

        self.segment_handle = self.env_config["params"]["Task"][
            "segment_handle"]
        self.segment_handle_camera_id = self.env_config["params"]["Task"][
            "segment_handle_camera_id"]
        self.use_fps = self.env_config["params"]["Task"]["use_fps"]

        self.aug_robot_mesh = self.env_config["params"]["Task"][
            "aug_robot_mesh"]
        if self.aug_robot_mesh:
            if not self.eval:
                save_robot_mesh(self.args_cli.log_dir, self.env)
            else:
                self.robot_mesh = load_robot_mesh(self.args_cli.log_dir, )

        if self.segment_handle:

            self.segment_handle_camera_id = self.segment_handle_camera_id
            self.sam_tool = ObjectSegmenter(
                self.device,
                box_threshold=0.25,
                text_threshold=0.25,
            )
            self.handle_mask = None

    def reset_data_buffer(self,
                          reset_cabinet=False,
                          reset_grasp=False,
                          reset_placement=False,
                          reset_close=False):
        if reset_cabinet:
            self.cabinet_obs_buffer = []
            self.cabinet_actions_buffer = []
            self.cabinet_rewards_buffer = []
            self.cabinet_does_buffer = []

        if reset_grasp:
            self.pick_obs_buffer = []
            self.pick_actions_buffer = []
            self.pick_rewards_buffer = []
            self.pick_does_buffer = []
        if reset_placement:

            self.place_obs_buffer = []
            self.place_actions_buffer = []
            self.place_rewards_buffer = []
            self.place_does_buffer = []
        if reset_close:
            self.close_obs_buffer = []
            self.close_actions_buffer = []
            self.close_rewards_buffer = []
            self.close_does_buffer = []

    def extract_data(self):

        if self.init_grasp:
            self.demo = self.collector_interface.grasp_raw_data["data"][
                f"demo_{self.demo_index}"]
            self.num_demos = len(
                self.collector_interface.grasp_raw_data["data"])

        if self.init_close:
            self.demo = self.collector_interface.close_raw_data["data"][
                f"demo_{self.demo_index}"]
            self.num_demos = len(
                self.collector_interface.close_raw_data["data"])

        if self.init_open:
            self.demo = self.collector_interface.cabinet_raw_data["data"][
                f"demo_{self.demo_index}"]
            self.num_demos = len(
                self.collector_interface.cabinet_raw_data["data"])

    def reset_rigid_articulation(self, target_name, pose_range):
        mdp.reset_root_state_uniform(self.env,
                                     env_ids=self.env_ids,
                                     pose_range=pose_range,
                                     velocity_range={},
                                     asset_cfg=SceneEntityCfg(target_name))

    def get_init_setting(self):
        if self.args_cli.eval_type == "train" or self.args_cli.eval_type == "replay":

            obs = self.demo["obs"]
            init_joint_pos = torch.as_tensor(obs["joint_pos"][0]).to(
                self.device).unsqueeze(0)
            robot_base = torch.as_tensor(obs["robot_base"][0]).to(
                self.device).unsqueeze(0)

            object_root_pose = torch.as_tensor(obs["mug_root_pose"][0]).to(
                self.device).unsqueeze(0)

            self.robot.write_root_pose_to_sim(robot_base, env_ids=self.env_ids)

            preset_object_root_states = self.grasp_object.data.default_root_state[
                self.env_ids].clone()
            if self.init_open:
                object_root_pose[:, 0] = 2.0

            self.grasp_object.write_root_pose_to_sim(object_root_pose,
                                                     env_ids=self.env_ids)
            self.grasp_object.write_root_velocity_to_sim(
                preset_object_root_states[:, 7:] * 0, env_ids=self.env_ids)
            return init_joint_pos
        elif self.args_cli.eval_type == "test":

            self.reset_rigid_articulation(
                "robot", self.robot_random_pose_range["pose_range"])
            self.reset_rigid_articulation(
                self.env.scene["kitchen"].cfg.
                articulation_cfg["target_object"],
                self.target_object_random_pose_range)
            obs = self.demo["obs"]
            init_joint_pos = torch.as_tensor(
                obs["joint_pos"][self.demo_index]).to(self.device).unsqueeze(0)

            return init_joint_pos

    def random_camera_pose(self):

        for key in self.env.scene.sensors:
            if "camera" in key:
                print("random camera")
                init_pose = self.env.scene.sensors[key].init_poses
                init_quat = self.env.scene.sensors[key].init_quats
                init_pose = init_pose + (torch.rand(
                    init_pose.shape, device=self.device) - 0.5) * 0.03
                init_orientation = torch.cat(
                    math_utils.euler_xyz_from_quat(init_quat)).unsqueeze(0)

                init_orientation += (
                    torch.rand(  # random orientation
                        init_orientation.shape,
                        device=self.device) - 0.5) * 0.02
                init_quat = math_utils.quat_from_euler_xyz(
                    init_orientation[:, 0], init_orientation[:, 1],
                    init_orientation[:, 2])

                self.env.scene.sensors[key]._view.set_world_poses(
                    init_pose, init_quat, self.env_ids)

    def reset_env(self):
        if self.segment_handle and self.handle_mask is None:

            for i in range(20):
                self.env.sim.step()

        self.env.reset()
        init_joint_pos = self.get_init_setting()
        default_jpos = self.kitchen._data.default_joint_pos.clone()

        # if self.random_camera and self.demo_index > 0:
        #     self.random_camera_pose()

        if (self.init_grasp or self.init_placement) and not self.init_close:

            self.kitchen._data.reset_joint_pos = default_jpos
            default_jpos[:, self.
                         joint_ids] = self.kitchen._data.joint_limits[:, self.
                                                                      joint_ids,
                                                                      -1]
            self.kitchen._data.reset_joint_pos = default_jpos
            self.kitchen.root_physx_view.set_dof_positions(
                default_jpos, self.env_ids)
        elif self.init_close:
            default_jpos[:, self.
                         joint_ids] = self.kitchen._data.joint_limits[:, self.
                                                                      joint_ids,
                                                                      0]

            self.kitchen._data.reset_joint_pos = default_jpos
            self.kitchen.root_physx_view.set_dof_positions(
                default_jpos, self.env_ids)
        else:
            self.kitchen._data.reset_joint_pos[:] = 0.0
            self.kitchen.root_physx_view.set_dof_positions(
                self.kitchen._data.reset_joint_pos, self.env_ids)

        obs = self.reset_step_env(init_joint_pos)
        if self.use_bounding_box:
            self.init_bbox = None
        if self.segment_handle:
            self.handle_points = self.sam_for_handle_segmentation(obs)

        return obs

    def reset_step_env(self, init_joint_pos):
        self.robot.root_physx_view.set_dof_positions(init_joint_pos[:, :9],
                                                     self.env_ids)

        for i in range(20):  # reset for stable initial statu9

            if not self.use_joint_pos:
                if self.use_relative_pose:
                    observation, reward, terminate, time_out, info = self.env.step(
                        torch.rand(self.env.action_space.shape,
                                   device=self.device) * 0.0)
                else:

                    observation, reward, terminate, time_out, info = self.env.step(
                        torch.as_tensor(self.demo["obs"]["ee_pose"][0]
                                        [:8]).unsqueeze(0).to(self.device))
            else:

                observation, reward, terminate, time_out, info = self.env.step(
                    torch.as_tensor(self.demo["obs"]["control_joint_action"]
                                    [3]).unsqueeze(0).to(self.device)[..., :8])

        return observation

    def sam_for_handle_segmentation(self, observation):

        pc = observation["policy"]["seg_pc"][0]
        rgb = observation["policy"]["rgb"][0]
        segmentation = observation["policy"]["segmentation"][0]
        id2label = observation["policy"]["id2lables"]
        drawer_ids = []
        valid_mask = []

        for i in range(pc.shape[0]):
            idToLabels = id2label[i]

            for key in idToLabels.keys():
                for target_seg_key in self.seg_target_name:
                    if target_seg_key in idToLabels[key]:
                        if "drawer" in idToLabels[key]:
                            drawer_ids.append(int(key))

            valid_mask.append(
                torch.isin(pc[i, :, -1].reshape(-1),
                           torch.as_tensor(drawer_ids[i]).to(self.device)))

        stack_id = torch.stack(valid_mask)

        valid_handle_masks = stack_id.reshape(stack_id.shape[0], rgb.shape[1],
                                              rgb.shape[2])
        filtered_masks_indices = [
            index for index, mask in enumerate(valid_mask)
            if mask.float().sum() > 1000
        ]

        # Initialize a combined mask with the same spatial shape as rgb
        combined_mask = torch.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2]),
                                    dtype=torch.bool,
                                    device=rgb.device)
        target_handle_mask = torch.zeros(
            (rgb.shape[0], rgb.shape[1], rgb.shape[2]),
            dtype=torch.bool,
            device=rgb.device)

        if self.handle_mask is None:

            # Combine masks based on filtered_masks_indices
            for index in filtered_masks_indices:

                if index != self.segment_handle_camera_id:
                    continue
                combined_mask[index] |= valid_handle_masks[
                    index]  # Accumulate all valid masks into one
                segmentation = torch.zeros((rgb.shape[1], rgb.shape[2], 3),
                                           dtype=torch.uint8,
                                           device=rgb.device)
                segmentation[combined_mask[index]] = rgb[index, ..., :3][
                    combined_mask[index]]

                masks, boxes, labels, scores = self.sam_tool.get_masks(
                    (segmentation.cpu().numpy() / 255).astype(np.float32),
                    "handle.")

                # masks, boxes, labels, scores = self.sam_tool.get_masks((rgb[index, ..., :3].cpu().numpy()), "handle")
                overlay_region = []
                overlay_region_count = []
                for mask in masks:
                    handle_mask = torch.as_tensor(mask).to(self.device)
                    mask_tensor = combined_mask[index]

                    overlap = torch.logical_and(
                        handle_mask, valid_handle_masks[index]).any()
                    if overlap:
                        overlap_count = torch.logical_and(
                            handle_mask, mask_tensor).sum().item()
                        overlay_region.append(handle_mask.bool())
                        overlay_region_count.append(overlap_count)

                    # cv2.imwrite("test.png", mask * 255)
                    # cv2.imwrite("rgb.png",
                    #             rgb[index, ..., :3].cpu().numpy()[:, :, ::-1])
                    # cv2.imwrite("seg.png",
                    #             segmentation.cpu().numpy()[:, :, ::-1])
                handle_id = np.argmin(overlay_region_count)
                combined_mask[index] = overlay_region[handle_id]

            self.handle_mask = combined_mask.reshape(pc.shape[0], -1)

            handle_pc = pc[self.handle_mask]
            self.raw_handle_pc = handle_pc.clone()
            o3d = vis_pc(handle_pc[:, :3].cpu().numpy(),
                         handle_pc[:, 3:6].cpu().numpy())
            visualize_pcd([o3d])
            del self.sam_tool

        else:
            # handle_pc = pc[self.handle_mask]
            handle_pc = self.raw_handle_pc.clone()

        return handle_pc

    def process_pc(self, observation):
        # if self.segment_handle:
        #     self.handle_points = self.sam_for_handle_segmentation(observation)

        if "seg_pc" in observation["policy"].keys():

            pc = observation["policy"]["seg_pc"][0]
            id2label = observation["policy"]["id2lables"]

            self.segmentation_label = []
            valid_mask = []

            for i in range(pc.shape[0]):
                idToLabels = id2label[i]
                self.segmentation_label.append([])
                for key in idToLabels.keys():
                    for target_seg_key in self.seg_target_name:
                        if target_seg_key in idToLabels[key]:

                            self.segmentation_label[i].append(int(key))

                valid_mask.append(
                    torch.isin(
                        pc[i, :, -1].reshape(-1),
                        torch.as_tensor(self.segmentation_label[i]).to(
                            self.device)))

            valid_mask = torch.cat(valid_mask)
            pc = pc.reshape(-1, pc.shape[-1])

            if self.use_bounding_box:
                if self.init_bbox is None:

                    target_pc = pc[valid_mask]

                    x_min, y_min, z_min = torch.min(target_pc[..., :3],
                                                    dim=0).values
                    x_max, y_max, z_max = torch.max(target_pc[..., :3],
                                                    dim=0).values
                    noise_min = -0.03  # Decrease for minimum values
                    noise_max = 0.03  # Increase for maximum values

                    # Apply noise to expand the bounding box
                    if self.bbox_range is not None:
                        bbox_range0 = torch.as_tensor(
                            self.bbox_range, ).clone()
                        self.init_bbox = torch.tensor([
                            torch.min(bbox_range0[0], x_min) +
                            torch.rand(1, device=self.device) * noise_min,
                            torch.min(bbox_range0[1], y_min) +
                            torch.rand(1, device=self.device) * noise_min,
                            torch.min(bbox_range0[2], z_min) +
                            torch.rand(1, device=self.device) * noise_min,
                            torch.max(bbox_range0[3], x_max) +
                            torch.rand(1, device=self.device) * noise_min,
                            torch.max(bbox_range0[4], y_max) +
                            torch.rand(1, device=self.device) * noise_min,
                            torch.max(bbox_range0[5], z_max) +
                            torch.rand(1, device=self.device) * noise_min,
                        ],
                                                      device=self.device)

                    else:
                        self.init_bbox = torch.tensor([
                            x_min +
                            torch.rand(1, device=self.device) * noise_min,
                            y_min +
                            torch.rand(1, device=self.device) * noise_min,
                            z_min +
                            torch.rand(1, device=self.device) * noise_min,
                            x_max +
                            torch.rand(1, device=self.device) * noise_max,
                            y_max +
                            torch.rand(1, device=self.device) * noise_max,
                            z_max +
                            torch.rand(1, device=self.device) * noise_max,
                        ],
                                                      device=self.device)
                valid_mask = ((pc[..., 0] >= self.init_bbox[0]) &
                              (pc[..., 0] <= self.init_bbox[3]) &
                              (pc[..., 1] >= self.init_bbox[1]) &
                              (pc[..., 1] <= self.init_bbox[4]) &
                              (pc[..., 2] >= self.init_bbox[2]) &
                              (pc[..., 2] <= self.init_bbox[5]))

            valid_mask = valid_mask.float(
            )  # Convert valid_mask to float for use with torch.multinomial

            if not self.eval and not self.use_fps:  # collect data not need to process
                indices = torch.multinomial(valid_mask,
                                            self.num_samples_points,
                                            replacement=True)
                observation["policy"]["seg_pc"] = pc[indices].unsqueeze(0)[
                    ..., :6]

            elif self.use_fps and self.eval:
                indices = torch.multinomial(valid_mask,
                                            30000,
                                            replacement=True)
                point_clouds = pc[indices].unsqueeze(0)

                batch_size, num_points, num_dims = point_clouds.shape
                flattened_points = point_clouds.view(
                    -1, num_dims)  # Shape: (71*10000, 3)

                # Create a batch tensor to indicate each point cloud's batch index
                batch = torch.repeat_interleave(torch.arange(batch_size),
                                                num_points).to(
                                                    flattened_points.device)
                ratio = self.num_samples_points / num_points

                # Apply farthest point sampling
                sampled_idx = fps(point_clouds[:, :, :3].reshape(-1, 3),
                                  batch,
                                  ratio=ratio,
                                  batch_size=batch_size)
                sampled_points = flattened_points[sampled_idx]
                sampled_points_per_cloud = sampled_points.size(0) // batch_size
                output = sampled_points.view(batch_size,
                                             sampled_points_per_cloud,
                                             num_dims)
                observation["policy"]["seg_pc"] = output
                torch.cuda.empty_cache()
            elif (self.use_fps and not self.eval
                  and not self.segment_handle) or (self.use_fps
                                                   and not self.eval):
                indices = torch.multinomial(valid_mask,
                                            30000,
                                            replacement=True)
                point_clouds = pc[indices].unsqueeze(0)
                observation["policy"]["seg_pc"] = point_clouds

            observation["policy"].pop("id2lables")

        if self.segment_handle:

            observation["policy"][
                "handle_points"] = self.handle_points.unsqueeze(0)
            if self.eval:
                observation["policy"]["seg_pc"] = torch.cat([
                    observation["policy"]["seg_pc"],
                    self.handle_points.unsqueeze(0)
                ],
                                                            dim=1)

        return observation

    def step_grasp_unnormalized_env(self, last_obs):

        init_object_pose = last_obs["policy"]["mug_pose"].clone()

        raw_actions = torch.as_tensor(np.array(self.demo["actions"]))

        normalized_action = raw_actions.clone()
        normalized_action[:, :3] = self.collector_interface.normalize(
            raw_actions[:, :3],
            self.collector_interface.action_stats["action"])
        last_obs = self.process_pc(last_obs)

        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
        # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
        # visualize_pcd([o3d_pc])

        for i in range(len(normalized_action)):

            unnormalized_action = normalized_action[i].clone()

            unnormalized_action[:3] = self.collector_interface.unnormalize(
                normalized_action[i][:3].clone(),
                self.collector_interface.action_stats["action"])

            unnormalized_action = unnormalized_action.unsqueeze(0)

            observation, reward, terminate, time_out, info = self.env.step(
                unnormalized_action.to(self.device))

            observation = self.process_pc(observation)

            if self.collect_data:
                self.pick_obs_buffer.append(last_obs["policy"])

                self.pick_actions_buffer.append(
                    normalized_action[i].unsqueeze(0))
                self.pick_rewards_buffer.append(reward)
                self.pick_does_buffer.append(terminate)

            last_obs = observation
            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # visualize_pcd([o3d_pc])
        last_object_pose = last_obs["policy"]["mug_pose"]

        self.demo_index += 1
        stop = False

        if self.collect_data:
            if not self.init_placement:
                if last_object_pose[0, 2] - init_object_pose[0, 2] > 0.10:
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.grasp_collector_interface,
                        self.pick_obs_buffer, self.pick_actions_buffer,
                        self.pick_rewards_buffer, self.pick_does_buffer)
            else:
                if abs(last_obs["policy"]["drawer_pose"][0, 2] -
                       last_object_pose[0, 2]
                       ) < 0.05:  # the object is placed on the drawer
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.grasp_collector_interface,
                        self.pick_obs_buffer, self.pick_actions_buffer,
                        self.pick_rewards_buffer, self.pick_does_buffer)

            self.reset_data_buffer(reset_grasp=True)
        return stop

    def step_cabinet_unnormalized_env(self, last_obs):

        init_object_pose = last_obs["policy"]["mug_pose"].clone()

        raw_actions = torch.as_tensor(np.array(self.demo["actions"]))
        normalized_action = raw_actions.clone()
        normalized_action[:, :3] = self.collector_interface.normalize(
            raw_actions[:, :3],
            self.collector_interface.action_stats["action"])

        last_obs = self.process_pc(last_obs)
        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()

        # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
        # visualize_pcd([o3d_pc])

        for i in range(len(normalized_action)):

            unnormalized_action = normalized_action[i].clone()

            unnormalized_action[:3] = self.collector_interface.unnormalize(
                normalized_action[i][:3].clone(),
                self.collector_interface.action_stats["action"])

            observation, reward, terminate, time_out, info = self.env.step(
                unnormalized_action.unsqueeze(0).to(self.device))
            observation = self.process_pc(observation)

            if self.collect_data:
                self.cabinet_obs_buffer.append(last_obs["policy"])

                self.cabinet_actions_buffer.append(
                    normalized_action[i].unsqueeze(0))
                self.cabinet_rewards_buffer.append(reward)
                self.cabinet_does_buffer.append(terminate)

            last_obs = observation

        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
        # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
        # visualize_pcd([o3d_pc])

        self.demo_index += 1
        stop = False
        if self.collect_data:
            if self.kitchen._data.joint_pos[0][self.joint_ids] > 0.25:
                stop = self.collector_interface.add_demonstraions_to_buffer(
                    self.collector_interface.cabinet_collector_interface,
                    self.cabinet_obs_buffer, self.cabinet_actions_buffer,
                    self.cabinet_rewards_buffer, self.cabinet_does_buffer)
            self.reset_data_buffer(reset_cabinet=True)
        return stop

    def step_unnormalized_env(self, last_obs, skip_frame=1):
        last_obs = self.process_pc(last_obs)
        init_object_pose = last_obs["policy"]["mug_pose"].clone()

        if not self.use_joint_pos:
            action_normailized_range = 3

            raw_actions = torch.as_tensor(np.array(self.demo["actions"]))

            normalized_action = raw_actions.clone()

            normalized_action[:, :3] = self.collector_interface.normalize(
                raw_actions[:, :3],
                self.collector_interface.action_stats["action"])
        else:

            raw_actions = torch.as_tensor(
                np.array(self.demo["obs"]["control_joint_action"]))[..., :8]
            raw_actions[:, -1] = torch.sign(raw_actions[:, -1] - 0.01)
            normalized_action = raw_actions.clone()
            action_normailized_range = normalized_action.shape[1]

            normalized_action = self.collector_interface.normalize(
                raw_actions, self.collector_interface.action_stats["action"])

        if self.init_grasp:
            normalized_action = torch.cat([
                normalized_action[:24:skip_frame], normalized_action[24:36],
                normalized_action[36::skip_frame]
            ],
                                          dim=0)
        elif self.init_open:
            normalized_action = torch.cat(
                [normalized_action[:40:skip_frame], normalized_action[40::]],
                dim=0)

        for i in range(0, len(normalized_action)):

            unnormalized_action = normalized_action[i].clone()

            unnormalized_action[:
                                action_normailized_range] = self.collector_interface.unnormalize(
                                    normalized_action[i]
                                    [:action_normailized_range].clone(), self.
                                    collector_interface.action_stats["action"])
            if self.use_joint_pos:

                unnormalized_action[-1] = torch.as_tensor(
                    np.array(self.demo["actions"])[i])[-1]

                observation, reward, terminate, time_out, info = self.env.step(
                    unnormalized_action.unsqueeze(0).to(self.device))

            else:
                unnormalized_action = unnormalized_action[..., :8]
                observation, reward, terminate, time_out, info = self.env.step(
                    unnormalized_action.unsqueeze(0).to(self.device))

            if self.collect_data:

                self.save_replay_data(last_obs, reward, terminate,
                                      normalized_action[i].unsqueeze(0))

            last_obs = observation
            last_obs = self.process_pc(last_obs)

            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # visualize_pcd([o3d_pc])
        stop = False

        if self.collect_data:
            if self.init_close:
                if True:  #self.kitchen._data.joint_pos[0][self.joint_ids] < 0.10:
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.close_collector_interface,
                        self.close_obs_buffer, self.close_actions_buffer,
                        self.close_rewards_buffer, self.close_does_buffer)
                self.reset_data_buffer(reset_close=True)
            elif self.init_open:

                if (self.target_joint_type == "prismatic"
                        and self.kitchen._data.joint_pos[0][self.joint_ids]
                        > 0.16) or (self.target_joint_type == "revolute"
                                    and abs(self.kitchen._data.joint_pos[0][
                                        self.joint_ids]) > 1.0):
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.cabinet_collector_interface,
                        self.cabinet_obs_buffer, self.cabinet_actions_buffer,
                        self.cabinet_rewards_buffer, self.cabinet_does_buffer)
                self.reset_data_buffer(reset_cabinet=True)
            elif self.init_grasp and not self.init_placement:
                if last_obs["policy"]["mug_pose"].clone()[
                        0, 2] - init_object_pose[0, 2] > 0.10:
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.grasp_collector_interface,
                        self.pick_obs_buffer, self.pick_actions_buffer,
                        self.pick_rewards_buffer, self.pick_does_buffer)
                self.reset_data_buffer(reset_grasp=True)
            elif self.init_placement:

                if abs(last_obs["policy"]["drawer_pose"][0, 2] -
                       last_obs["policy"]["mug_pose"][0, 2]) < 0.07:
                    stop = self.collector_interface.add_demonstraions_to_buffer(
                        self.collector_interface.grasp_collector_interface,
                        self.pick_obs_buffer, self.pick_actions_buffer,
                        self.pick_rewards_buffer, self.pick_does_buffer)
                self.reset_data_buffer(reset_grasp=True)

        self.demo_index += 1

        return stop

    def process_obs(self, last_obs, key):
        if key == "seg_pc":
            per_obs = last_obs["policy"][key].permute(0, 2, 1)[:, :3, :]
        else:
            per_obs = last_obs["policy"][key]
        return per_obs

    def eval_env(self, last_obs, stats, normalized_grasp, open_loop=False):
        # self.load_policy(self.policy_path)
        last_obs = self.process_pc(last_obs)
        # create obs buffer
        obs_buffer = {}
        gt_actions = torch.as_tensor(
            np.array(normalized_grasp["data"][f"demo_{self.demo_index}"]
                     ["actions"]))  #to(self.device)

        obs = normalized_grasp["data"][f"demo_{self.demo_index}"]["obs"]

        print(obs["mug_root_pose"][0][:3],
              last_obs["policy"]["mug_root_pose"][..., :3])

        init_object_pose = last_obs["policy"]["mug_pose"].clone()
        if not open_loop:
            for key in last_obs["policy"].keys():
                per_obs = self.process_obs(last_obs, key)

                obs_buffer[key] = torch.cat([per_obs, per_obs], dim=0)
        else:
            for key in obs.keys():
                obs_buffer[key] = torch.cat([
                    torch.as_tensor(obs[key][0]).unsqueeze(0),
                    torch.as_tensor(obs[key][0]).unsqueeze(0)
                ],
                                            dim=0).to(self.device)
        # obs_buffer["seg_pc"] = obs_buffer["seg_pc"].permute(0, 2, 1)[:, :3, :]
        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
        # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
        # visualize_pcd([o3d_pc])

        # seg_pc = np.array(obs["seg_pc"][0])
        # o3d_pc = vis_pc(seg_pc[:, :3], None)
        # visualize_pcd([o3d_pc])
        self.demo_index += 8
        if not open_loop:
            horizon = len(gt_actions) + 50
        else:
            horizon = len(gt_actions)

        for actions_id in range(horizon):

            if not open_loop:
                for key in last_obs["policy"].keys():
                    obs_buffer[key][0] = obs_buffer[key][-1].clone()
                    per_obs = self.process_obs(last_obs, key)
                    obs_buffer[key][-1] = per_obs
                    predicted_action = self.policy(obs_buffer)
            else:
                for key in obs.keys():
                    obs_buffer[key][0] = obs_buffer[key][-1].clone()

                    obs_buffer[key][-1] = torch.as_tensor(
                        obs[key][actions_id]).to(self.device).unsqueeze(0)
                    target_obs = copy.deepcopy(obs_buffer)
                    target_obs["seg_pc"] = target_obs["seg_pc"].permute(
                        0, 2, 1)[:, :3, :]

                    predicted_action = self.policy(target_obs)

            # last_seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # seg_pc = obs["seg_pc"][actions_id]

            # o3d_pc = vis_pc(
            #     np.concatenate([
            #         seg_pc[:, :3],
            #         last_seg_pc[:, :3],
            #     ], axis=0),
            #     np.concatenate([
            #         seg_pc[:, 3:6],
            #         last_seg_pc[:, 3:6] * 0.0,
            #     ],
            #                    axis=0))
            # visualize_pcd([o3d_pc])

            # predicted_action = torch.as_tensor(gt_actions[actions_id])

            predicted_action[:3] = self.collector_interface.unnormalize(
                predicted_action[:3][None],
                stats.item()["action"])[0]
            predicted_action = torch.as_tensor(predicted_action).unsqueeze(
                0).to(self.device)

            predicted_action[:, -1] = torch.sign(predicted_action[:, -1])

            observation, reward, terminate, time_out, info = self.env.step(
                predicted_action)
            observation = self.process_pc(observation)

            last_obs = observation
            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # visualize_pcd([o3d_pc])

            if self.init_grasp and last_obs["policy"]["mug_pose"].clone()[
                    0, 2] - init_object_pose[0, 2] > 0.15:
                if self.collect_data:
                    self.clear_eval_data()
                return True
            if self.init_close and self.kitchen._data.joint_pos[0][
                    self.joint_ids] < 0.05:
                return True
            if self.init_open and self.kitchen._data.joint_pos[0][
                    self.joint_ids] > 0.20:
                if self.collect_data:
                    self.clear_eval_data()
                return True
            if self.collect_data:
                self.save_eval_data(observation, reward, terminate,
                                    predicted_action)

        if self.collect_data:
            self.clear_eval_data()

        if self.init_grasp:
            return last_obs["policy"]["mug_pose"].clone()[
                0, 2] - init_object_pose[0, 2] > 0.15
        elif self.init_close:
            return self.kitchen._data.joint_pos[0][self.joint_ids] < 0.05
        elif self.init_open and self.kitchen._data.joint_pos[0][
                self.joint_ids] > 0.20:
            return True

    def clear_eval_data(self, ):
        if self.init_grasp:

            self.collector_interface.add_demonstraions_to_buffer(
                self.collector_interface.grasp_collector_interface,
                self.pick_obs_buffer, self.pick_actions_buffer,
                self.pick_rewards_buffer, self.pick_does_buffer)
            self.reset_data_buffer(reset_grasp=True)
        if self.init_open:
            self.collector_interface.add_demonstraions_to_buffer(
                self.collector_interface.cabinet_collector_interface,
                self.cabinet_obs_buffer, self.cabinet_actions_buffer,
                self.cabinet_rewards_buffer, self.cabinet_does_buffer)
            del self.cabinet_actions_buffer, self.cabinet_obs_buffer, self.cabinet_rewards_buffer, self.cabinet_does_buffer
            self.reset_data_buffer(reset_cabinet=True)

    def save_eval_data(self, observation, reward, terminate, predicted_action):

        keys_to_remove = [
            key for key in observation["policy"].keys()
            if key in self.filter_keys
        ]
        for key in keys_to_remove:
            observation["policy"].pop(
                key, None
            )  # Use pop with default to avoid KeyError if key is missing
        obs_buffer = {}

        for key in observation["policy"].keys():

            if key == "id2lables":
                continue

            obs_buffer[key] = observation["policy"][key].cpu()
        if self.init_grasp:
            self.pick_obs_buffer.append(obs_buffer)
            self.pick_actions_buffer.append(predicted_action)
            self.pick_rewards_buffer.append(reward)
            self.pick_does_buffer.append(terminate)
        if self.init_open:

            self.cabinet_obs_buffer.append(obs_buffer)
            self.cabinet_actions_buffer.append(predicted_action)
            self.cabinet_rewards_buffer.append(reward)
            self.cabinet_does_buffer.append(terminate)
        if self.init_close:

            self.close_obs_buffer.append(obs_buffer)
            self.close_actions_buffer.append(predicted_action)
            self.close_rewards_buffer.append(reward)
            self.close_does_buffer.append(terminate)

    def save_replay_data(self, observation, reward, terminate,
                         predicted_action):

        if self.init_grasp:
            self.pick_obs_buffer.append(observation["policy"])
            self.pick_actions_buffer.append(predicted_action)
            self.pick_rewards_buffer.append(reward)
            self.pick_does_buffer.append(terminate)
        if self.init_open:

            self.cabinet_obs_buffer.append(observation["policy"])
            self.cabinet_actions_buffer.append(predicted_action)
            self.cabinet_rewards_buffer.append(reward)
            self.cabinet_does_buffer.append(terminate)
        if self.init_close:

            self.close_obs_buffer.append(observation["policy"])
            self.close_actions_buffer.append(predicted_action)
            self.close_rewards_buffer.append(reward)
            self.close_does_buffer.append(terminate)

    def eval_open_loop_env(self, last_obs):
        # create obs buffer
        obs_buffer = {}
        gt_actions = torch.as_tensor(np.array(self.demo["actions"])).to(
            self.device)

        all_obs = self.demo["obs"]
        for key in all_obs.keys():

            obs_buffer[key] = torch.cat([
                torch.as_tensor(all_obs[key][0]).unsqueeze(0),
                torch.as_tensor(all_obs[key][0]).unsqueeze(0)
            ],
                                        dim=0).to(self.device)
        for actions_id in range(len(gt_actions)):
            for key in all_obs.keys():
                obs_buffer[key][0] = obs_buffer[key][-1].clone()

                obs_buffer[key][-1] = torch.as_tensor(
                    all_obs[key][actions_id]).to(self.device).unsqueeze(0)
            predicted_action = self.policy(obs_buffer)
            predicted_action = torch.as_tensor(predicted_action).unsqueeze(
                0).to(self.device)

            predicted_action[:, -1] = torch.sign(predicted_action[:, -1])

            observation, reward, terminate, time_out, info = self.env.step(
                predicted_action)

            last_obs = observation

    def replay_normalized_action(self, ):
        self.env.reset()
        init_joint_pos = self.get_init_setting()

        object_root_pose = torch.as_tensor(
            self.demo['obs']["mug_root_pose"][0]).to(self.device).unsqueeze(0)

        self.grasp_object.write_root_pose_to_sim(object_root_pose,
                                                 env_ids=self.env_ids)
        self.grasp_object.write_root_velocity_to_sim(object_root_pose[:, :6] *
                                                     0,
                                                     env_ids=self.env_ids)
        self.reset_step_env(init_joint_pos)

        actions = np.array(self.demo["actions"])
        if not self.use_joint_pos:
            action_dim = 3
        else:
            action_dim = 8

        for i in range(0, len(actions)):
            unnormalized_action = torch.as_tensor(actions[i]).clone()

            unnormalized_action[:
                                action_dim] = self.collector_interface.unnormalize(
                                    unnormalized_action[:action_dim].clone(),
                                    self.collector_interface.
                                    action_stats["action"])

            observation, reward, terminate, time_out, info = self.env.step(
                torch.as_tensor(unnormalized_action).unsqueeze(0).to(
                    self.device))

    def eval_bc_env(self, last_obs, open_loop=False):
        self.policy.start_episode()

        last_obs = self.process_pc(last_obs)
        init_object_pose = last_obs["policy"]["mug_pose"].clone()
        # obs = self.demo["obs"]
        # gtaction = self.demo["actions"]

        if not open_loop:
            horizon = 100  #len(gtaction) + 30
        else:
            horizon = 100  # len(gtaction)

        for actions_id in range(horizon):

            if not open_loop:

                input_obs = copy.deepcopy(last_obs["policy"])

                input_obs["seg_pc"] = input_obs["seg_pc"].permute(0, 2,
                                                                  1)[0, :3, :]

                input_obs["ee_pose"] = torch.cat([
                    input_obs["ee_pose"],
                    torch.as_tensor([np.clip(actions_id, 0, 100)]).to(
                        self.device).unsqueeze(-1)
                ],
                                                 dim=-1)

                if self.aug_robot_mesh:
                    aug_robot_points = aug_robot_mesh(
                        last_obs["policy"]["robot_link_pose"].cpu(),
                        self.robot_mesh).to(self.device)

                    input_obs["seg_pc"] = torch.cat([
                        input_obs["seg_pc"],
                        aug_robot_points.permute(0, 2, 1)[0, :3, :]
                    ],
                                                    dim=1)
                predicted_action = self.policy(input_obs)
            # else:
            #     index_obs = {}
            #     for key in obs.keys():
            #         index_obs[key] = torch.as_tensor(obs[key][actions_id]).to(
            #             self.device).unsqueeze(0)

            #     index_obs["seg_pc"] = index_obs["seg_pc"].permute(0, 2,
            #                                                       1)[0, :3, :]

            #     predicted_action = self.policy(index_obs)

            predicted_action[:3] = self.collector_interface.unnormalize(
                predicted_action[:3][None],
                self.collector_interface.action_stats["action"])[0]

            predicted_action = torch.as_tensor(predicted_action).unsqueeze(
                0).to(self.device)

            predicted_action[:, -1] = torch.sign(predicted_action[:, -1])

            observation, reward, terminate, time_out, info = self.env.step(
                predicted_action)

            last_obs = observation
            last_obs = self.process_pc(last_obs)

        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()

        # o3d_pc = vis_pc(seg_pc[:, :3], None)
        # o3d_pc2 = vis_pc(aug_robot_points[0][:, :3].cpu().numpy(), None)
        # visualize_pcd([o3d_pc, o3d_pc2])

        self.demo_index += 5
        if self.init_grasp and not self.init_placement:
            return last_obs["policy"]["mug_pose"].clone()[
                0, 2] - init_object_pose[0, 2] > 0.10
        elif self.init_placement and self.init_grasp:
            return abs(last_obs["policy"]["drawer_pose"][0, 2] -
                       last_obs["policy"]["mug_pose"][0, 2]) < 0.07
        elif self.init_close:
            return self.kitchen._data.joint_pos[0][self.joint_ids] < 0.05
        elif self.init_open and (
            (self.target_joint_type == "prismatic"
             and self.kitchen._data.joint_pos[0][self.joint_ids] > 0.16) or
            (self.target_joint_type == "revolute"
             and abs(self.kitchen._data.joint_pos[0][self.joint_ids]) > 1.0)):
            return True

    def eval_3df_env(self, last_obs, open_loop=False, state_key="joint_pos"):
        # self.load_policy(self.policy_path)

        rollout_actions = []

        last_obs = self.process_pc(last_obs)

        # create obs buffer
        obs_buffer = {}
        gt_actions = torch.as_tensor(np.array(
            self.demo["actions"]))  #to(self.device)

        obs = self.demo["obs"]
        # seg_pcs = obs["seg_pc"]

        # handle_points = np.array(obs["handle_points"])
        # del obs["seg_pc"]
        # obs["seg_pc"] = np.concatenate([seg_pcs[...,:3], handle_points[...,:3]], axis=1)

        print(obs["mug_root_pose"][0][:3],
              last_obs["policy"]["mug_root_pose"][..., :3])

        init_object_pose = last_obs["policy"]["mug_pose"].clone()
        # seg_pc = obs["seg_pc"][0]
        # o3d_pc = vis_pc(seg_pc[:, :3], None)
        # visualize_pcd([o3d_pc])

        if not open_loop:

            obs_buffer["point_cloud"] = torch.cat(
                [last_obs["policy"]["seg_pc"], last_obs["policy"]["seg_pc"]],
                dim=0).unsqueeze(0)[..., :3]
            obs_buffer["agent_pos"] = torch.cat(
                [last_obs["policy"][state_key], last_obs["policy"][state_key]],
                dim=0).unsqueeze(0)

            obs_buffer["agent_pos"][0][-1] = torch.as_tensor(
                last_obs["policy"][state_key][0]).to(self.device)
            # obs_buffer["agent_pos"][0][-1] = torch.as_tensor(
            #     obs[state_key][0]).to(self.device)[..., :8]

            if self.aug_robot_mesh:
                aug_robot_points = aug_robot_mesh(
                    last_obs["policy"]["robot_link_pose"].cpu(),
                    self.robot_mesh).to(self.device)
                obs_buffer["imagin_robot"] = torch.cat(
                    [aug_robot_points, aug_robot_points], dim=0).unsqueeze(0)

        else:

            obs_buffer["point_cloud"] = torch.cat(
                [
                    torch.as_tensor(obs["seg_pc"][0]).unsqueeze(0),
                    torch.as_tensor(obs["seg_pc"][0]).unsqueeze(0)
                ],
                dim=0).to(self.device).unsqueeze(0)[..., :3]
            obs_buffer["agent_pos"] = torch.cat([
                torch.as_tensor(obs[state_key][0]).unsqueeze(0),
                torch.as_tensor(obs[state_key][0]).unsqueeze(0)
            ],
                                                dim=0).to(
                                                    self.device).unsqueeze(0)
            if self.aug_robot_mesh:

                obs_buffer["imagin_robot"] = torch.cat(
                    [
                        torch.as_tensor(obs["imagin_robot"][0]).unsqueeze(0),
                        torch.as_tensor(obs["imagin_robot"][0]).unsqueeze(0)
                    ],
                    dim=0).to(self.device).unsqueeze(0)

        self.demo_index += 5
        if not open_loop:
            print("close loop evaluation")
            horizon = int(len(gt_actions)) + 20
        else:
            print("open loop evaluation")
            horizon = len(gt_actions)

        for actions_id in range(horizon):

            if not open_loop:
                #
                obs_buffer["point_cloud"][0][0] = obs_buffer["point_cloud"][0][
                    -1].clone()[..., :3] * 0.0
                obs_buffer["point_cloud"][0][-1] = last_obs["policy"][
                    "seg_pc"][0].to(self.device)[..., :3] * 0.0
                obs_buffer["agent_pos"][0][0] = obs_buffer["agent_pos"][0][
                    -1].clone()

                obs_buffer["agent_pos"][0][
                    -1] = last_obs["policy"][state_key][0].to(
                        self.device) + torch.randn(1, 8).to(self.device) * 0.10
                # obs_buffer["agent_pos"][0][-1] = torch.as_tensor(
                #     obs[state_key][actions_id]).to(self.device)[..., :-1]

                if self.aug_robot_mesh:
                    aug_robot_points = aug_robot_mesh(
                        last_obs["policy"]["robot_link_pose"].cpu(),
                        self.robot_mesh).to(self.device)
                    obs_buffer["imagin_robot"][0][0] = obs_buffer[
                        "imagin_robot"][0][-1].clone()
                    obs_buffer["imagin_robot"][0][-1] = aug_robot_points[0]

            else:

                obs_buffer["point_cloud"][0][0] = obs_buffer["point_cloud"][0][
                    -1].clone()
                obs_buffer["point_cloud"][0][-1] = torch.as_tensor(
                    obs["seg_pc"][actions_id]).to(self.device)[..., :3]

                obs_buffer["agent_pos"][0][0] = obs_buffer["agent_pos"][0][
                    -1].clone()
                obs_buffer["agent_pos"][0][-1] = torch.as_tensor(
                    obs[state_key][actions_id]).to(self.device)

                if self.aug_robot_mesh:
                    obs_buffer["imagin_robot"][0][0] = obs_buffer[
                        "imagin_robot"][0][-1].clone()
                    obs_buffer["imagin_robot"][0][-1] = torch.as_tensor(
                        obs["imagin_robot"][actions_id]).to(self.device)

            time_step_buffer = copy.deepcopy(obs_buffer)

            if not open_loop:

                time_step_buffer["agent_pos"] = torch.cat([
                    obs_buffer["agent_pos"],
                    torch.as_tensor([[
                        np.clip(actions_id - 1, 0, 100),
                        np.clip(actions_id, 0, 100)
                    ]]).to(self.device).unsqueeze(-1)
                ],
                                                          dim=-1)

            predicted_action = self.policy.predict_action(
                time_step_buffer)["action"][0][0].cpu()

            # rollout_actions.append(predicted_action.cpu().numpy())
            # print(predicted_action[-1])

            if not self.use_joint_pos:
                predicted_action[:3] = self.collector_interface.unnormalize(
                    predicted_action[:3][None],
                    self.collector_interface.action_stats["action"])[0]
            else:

                predicted_action[:8] = self.collector_interface.unnormalize(
                    predicted_action[:8][None],
                    self.collector_interface.action_stats["action"])[0]
            predicted_action = torch.as_tensor(predicted_action).unsqueeze(
                0).to(self.device)
            # print(predicted_action[:, -1])
            #

            predicted_action[:, -1] = torch.sign(predicted_action[:, -1])
            # predicted_action[:, 1] -= 0.02

            observation, reward, terminate, time_out, info = self.env.step(
                predicted_action)
            # print(
            #     torch.linalg.norm(observation["policy"]["ee_pose"][:, :3] -
            #                       predicted_action[:, :3]))

            observation = self.process_pc(observation)

            last_obs = copy.deepcopy(observation)

            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # o3d_pc2 = vis_pc(aug_robot_points[0][:, :3].cpu().numpy(), None)
            # visualize_pcd([o3d_pc, o3d_pc2])

            # if self.init_grasp and not self.init_placement and last_obs[
            #         "policy"]["mug_pose"].clone()[0, 2] - init_object_pose[
            #             0, 2] > 1.15:
            #     if self.collect_data:
            #         self.clear_eval_data()
            #     return True
            # elif self.init_placement and self.init_grasp and abs(
            #         last_obs["policy"]["drawer_pose"][0, 2] -
            #         last_obs["policy"]["mug_pose"][0, 2]) < 0.05:
            #     if self.collect_data:
            #         self.clear_eval_data()
            #     return True

            # if self.init_close and self.kitchen._data.joint_pos[0][
            #         self.joint_ids] < 0.05:
            #     return True
            # if self.init_open and self.kitchen._data.joint_pos[0][
            #         self.joint_ids] > 0.20:
            #     if self.collect_data:
            #         self.clear_eval_data()
            #     return True
            # if self.collect_data:
            #     self.save_eval_data(observation, reward, terminate,
            #                         predicted_action)
        print("robot_tip", get_robottip_pose(self.robot, self.device)[0])
        observation["policy"]["ee_pose"]
        # import pdb
        # pdb.set_trace()
        # import pdb

        # pdb.set_trace()
        # np.concatenate(rollout_actions, axis=0)

        # np.save("rollout_actions.npy", rollout_actions)

        # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
        # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
        # visualize_pcd([o3d_pc])

        if self.collect_data:
            self.clear_eval_data()

        if self.init_grasp and not self.init_placement:
            return last_obs["policy"]["mug_pose"].clone()[
                0, 2] - init_object_pose[0, 2] > 0.10
        elif self.init_placement and self.init_grasp:
            return abs(last_obs["policy"]["drawer_pose"][0, 2] -
                       last_obs["policy"]["mug_pose"][0, 2]) < 0.07
        elif self.init_close:
            return self.kitchen._data.joint_pos[0][self.joint_ids] < 0.05
        elif self.init_open and (
                self.target_joint_type == "prismatic"
                and self.kitchen._data.joint_pos[0][self.joint_ids]
                > 0.16) or (self.target_joint_type == "revolute" and abs(
                    self.kitchen._data.joint_pos[0][self.joint_ids]) > 1.0):
            return True
