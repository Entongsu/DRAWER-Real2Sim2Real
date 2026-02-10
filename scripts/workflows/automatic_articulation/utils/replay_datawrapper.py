from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from tools.visualization_utils import vis_pc, visualize_pcd

from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose
import cv2
import sys

try:
    sys.path.append("/media/lme/data4/weird/Grounded-SAM-2")
    from sam_tool import ObjectSegmenter
except:
    pass
import json
import pytorch3d.ops as torch3d_ops
# from torch_cluster import fps
# from dgl.geometry import farthest_point_sampler

from omegaconf import OmegaConf
from tools.visualization_utils import *
from scripts.workflows.automatic_articulation.utils.map_env import step_buffer_map, reset_buffer_map, reset_data_buffer, init_setting
from scripts.workflows.automatic_articulation.utils.load_obj_utils import save_robot_mesh
from scripts.workflows.automatic_articulation.task.multi_step_env import MultiStepEnv


class ReplayDatawrapper:

    def __init__(
        self,
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
        num_samples_points=2048,
        eval=False,
        replay_normalized_actions=False,
        add_action_noise=False,
        action_noise=[0, 0],
        eval_3d_diffusion=False,
        use_joint_pos=False,
        env_config=None,
    ):
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
        self.env_config = env_config

        self.eval = eval

        self.replay_normalized_actions = replay_normalized_actions
        self.action_noise = action_noise
        self.eval_3d_diffusion = eval_3d_diffusion

        self.use_joint_pos = use_joint_pos

        self.filter_keys = filter_keys
        self.init_setting(args_cli, init_grasp, init_open, init_placement,
                          init_close, filter_keys, collect_data)

        self.multi_env = MultiStepEnv(env,
                                      collision_checker=False,
                                      use_relative_pose=use_relative_pose,
                                      init_grasp=init_grasp,
                                      init_open=init_open,
                                      init_placement=init_placement,
                                      init_close=init_close,
                                      env_config=env_config)
        self.demo_index = 0

    def init_setting(self, args_cli, init_grasp, init_open, init_placement,
                     init_close, filter_keys, collect_data):

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

        if collect_data:
            reset_data_buffer(self,
                              reset_cabinet=True,
                              reset_grasp=True,
                              reset_placement=True,
                              reset_close=True)

    def load_config(self):

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

    def reset_demo_env(self):

        obs = self.demo["obs"]

        # extract init robot setting
        init_joint_pos = torch.as_tensor(obs["joint_pos"][0]).to(
            self.device).unsqueeze(0)
        robot_base = torch.as_tensor(obs["robot_base"][0]).to(
            self.device).unsqueeze(0)
        self.multi_env.robot.write_root_pose_to_sim(
            robot_base, env_ids=self.multi_env.env_ids)

        # extract init object setting
        object_root_pose = torch.as_tensor(obs["mug_root_pose"][0]).to(
            self.device).unsqueeze(0)
        preset_object_root_states = self.env.scene[
            self.multi_env.grasp_object].data.default_root_state[
                self.multi_env.env_ids].clone()
        self.env.scene[self.multi_env.grasp_object].write_root_pose_to_sim(
            object_root_pose, env_ids=self.multi_env.env_ids)
        self.env.scene[self.multi_env.grasp_object].write_root_velocity_to_sim(
            preset_object_root_states[:, 7:] * 0,
            env_ids=self.multi_env.env_ids)
        # set the kitchen drawer to the initial state
        self.multi_env.reset_kitchen_drawer(
            init_grasp=False if self.init_open else True)

        # reset env
        self.multi_env.robot.root_physx_view.set_dof_positions(
            init_joint_pos[:, :9], self.multi_env.env_ids)

        if self.init_grasp:
            self.multi_env.env_grasp.init_grasp_object_state = self.env.scene[
                self.multi_env.grasp_object]._data.root_state_w[:, :3]

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

    def reset_env(self):

        if self.segment_handle and self.handle_mask is None:

            for i in range(20):
                self.env.sim.step()

        self.env.reset()
        if self.args_cli.eval_type == "train" or self.args_cli.eval_type == "replay":
            self.extract_data()
            obs = self.reset_demo_env()
        else:
            obs = None
            while obs is None:
                obs = self.multi_env.reset_all_env(
                    reset_grasp=self.init_grasp,
                    reset_cabinet=self.init_open,
                    reset_close=self.init_close,
                )

        if self.use_bounding_box:
            self.init_bbox = None
        if self.segment_handle:
            self.handle_points = self.sam_for_handle_segmentation(obs)

        return obs

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

    def update_success_flags(self):
        # Mapping init flags to corresponding success flags
        success_map = {
            'init_open': 'cabinet_success',
            'init_close': 'close_success',
            'init_grasp': 'grasp_success',
            'init_placement': 'placement_success'
        }

        # Update success flags based on init flags
        for init_flag, success_flag in success_map.items():
            setattr(self, success_flag, getattr(self, init_flag, False))

    def step_unnormalized_env(self, last_obs, skip_frame=1):

        last_obs = self.process_pc(last_obs)
        init_object_pose = last_obs["policy"]["mug_pose"].clone()
        collect_cabinet, collect_close, collect_grasp, collect_placement = self.init_open, self.init_close, self.init_grasp, self.init_placement

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
            print(i, len(normalized_action))

            unnormalized_action = normalized_action[i].clone()

            unnormalized_action[:
                                action_normailized_range] = self.collector_interface.unnormalize(
                                    normalized_action[i]
                                    [:action_normailized_range].clone(), self.
                                    collector_interface.action_stats["action"])
            if self.use_joint_pos:

                unnormalized_action[-1] = torch.as_tensor(
                    np.array(self.demo["actions"])[i])[-1]

            else:
                unnormalized_action = unnormalized_action[..., :8]

            observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                unnormalized_action,
                collect_grasp=self.init_grasp,
                collect_placement=self.init_placement,
                collect_cabinet=self.init_open,
                collect_close=self.init_close)

            if self.collect_data:

                for flag, buffers in step_buffer_map.items():
                    if locals(
                    )[flag]:  # Dynamically check if the corresponding flag is True
                        getattr(self, buffers[0]).append(
                            last_obs["policy"])  # Observation
                        getattr(self, buffers[1]).append(actions)  # Actions
                        getattr(self, buffers[2]).append(reward)  # Rewards
                        getattr(self, buffers[3]).append(
                            terminate)  # Termination status

            last_obs = observation
            last_obs = self.process_pc(last_obs)

            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # visualize_pcd([o3d_pc])
        stop = False

        if self.collect_data:
            if success:
                self.update_success_flags()
                self.cache_data()
            reset_data_buffer(self,
                              reset_grasp=True,
                              reset_cabinet=True,
                              reset_close=True,
                              reset_placement=True)

        self.demo_index += 1

        return stop

    def cache_data(self):
        # Map success flags to their corresponding interface strings and buffers

        stop_collect = False

        # Iterate through the map and cache data if the success flag is True
        for flag, (interface_attr, buffers) in collect_map.items():

            if getattr(self, flag):  # Dynamically check the success flag
                # Dynamically get the interface attribute from collector_interface
                interface = getattr(self.collector_interface, interface_attr,
                                    None)
                if interface is None:
                    continue

                # Add demonstrations to buffer
                stop_collect = self.collector_interface.add_demonstraions_to_buffer(
                    interface,
                    getattr(self, buffers[0]),  # Observation buffer
                    getattr(self, buffers[1]),  # Actions buffer
                    getattr(self, buffers[2]),  # Rewards buffer
                    getattr(self, buffers[3])  # Termination status buffer
                )

        # Reset all buffers after caching
        reset_data_buffer(self,
                          reset_close=True,
                          reset_cabinet=True,
                          reset_grasp=True,
                          reset_placement=True)
        return stop_collect

    def process_obs(self, last_obs, key):
        if key == "seg_pc":
            per_obs = last_obs["policy"][key].permute(0, 2, 1)[:, :3, :]
        else:
            per_obs = last_obs["policy"][key]
        return per_obs
