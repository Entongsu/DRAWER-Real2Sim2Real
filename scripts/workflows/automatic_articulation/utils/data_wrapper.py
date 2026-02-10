from scripts.workflows.automatic_articulation.task.multi_step_env import MultiStepEnv
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.automatic_articulation.utils.map_env import step_buffer_map, reset_data_buffer, load_config, placement_region
import torch
import numpy as np
import copy
import open3d as o3d
import pymeshlab
import isaaclab.utils.math as math_utils
from tools.visualization_utils import vis_pc, visualize_pcd
import sys
try:
    sys.path.append("/home/ensu/Documents/weird/Grounded-SAM-2")
    from sam_tool import ObjectSegmenter
except:
    pass


class Datawrapper:

    def __init__(
        self,
        env,
        collision_checker,
        use_relative_pose,
        init_grasp,
        init_open,
        init_placement,
        init_close,
        collect_data=None,
        args_cli=None,
        env_config=None,
        filter_keys=None,
        load_path=None,
        save_path=None,
        use_fps=False,
        use_joint_pos=False,
        use_demo_data=False,
        seg_handle=False,
    ):
        self.multi_env = MultiStepEnv(env,
                                      collision_checker=collision_checker,
                                      use_relative_pose=use_relative_pose,
                                      init_grasp=init_grasp,
                                      init_open=init_open,
                                      init_placement=init_placement,
                                      init_close=init_close,
                                      env_config=env_config)
        self.env = env
        self.init_open = init_open
        self.init_grasp = init_grasp
        self.init_placement = init_placement
        self.init_close = init_close

        self.grasp_success = False
        self.close_success = False
        self.cabinet_success = False

        self.collector_interface = None
        self.reset_all = True
        self.env_config = env_config
        self.use_demo_data = use_demo_data

        self.grasp_success_count = 0
        self.demo_index = 0
        self.args_cli = args_cli
        self.use_joint_pos = use_joint_pos
        self.seg_handle = seg_handle
        if self.seg_handle:
            self.segmentation_for_handle()
        load_config(self)

        # if collect_data:
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            filter_keys,
            load_path=load_path,
            save_path=save_path,
            use_fps=use_fps,
            use_joint_pos=use_joint_pos)
        self.collector_interface.init_collector_interface()
        if collect_data:
            reset_data_buffer(self,
                              reset_cabinet=True,
                              reset_grasp=True,
                              reset_placement=True,
                              reset_close=True)
        self.handle_mask = None

    def segmentation_for_handle(self, ):

        self.target_handle_name = self.env_config["params"][
            "ArticulationObject"]["kitchen"]["target_drawer"]
        meta_data = self.multi_env.planner.obstacles_mesh[
            self.target_handle_name]
        vertices = np.array(meta_data[1].tolist())
        faces = np.array(meta_data[2].tolist())
        while vertices.shape[0] < 1000:
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            mesh = pymeshlab.Mesh(vertex_matrix=vertices,
                                  face_matrix=np.array(faces, dtype=np.int32))

            # Create a MeshSet and add the mesh to it
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh, 'my_mesh')
            ms.meshing_remove_duplicate_faces()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_surface_subdivision_midpoint(iterations=5)
            current_mesh = ms.current_mesh()
            vertices = current_mesh.vertex_matrix()
            vertices = torch.tensor(vertices,
                                    dtype=torch.float32).to(self.env.device)
        self.handle_vertices = vertices
        self.sam_tool = ObjectSegmenter(
            self.env.device,
            box_threshold=0.25,
            text_threshold=0.25,
        )

    def sam_for_handle_segmentation(self, observation):

        pc = observation["policy"]["whole_pc"][0]
        rgb = observation["policy"]["rgb"][0]
        segmentation = observation["policy"]["segmentation"][0]
        id2label = observation["policy"]["id2lables"]
        drawer_ids = []
        valid_mask = []

        for i in range(pc.shape[0]):
            idToLabels = id2label[i]

            for key in idToLabels.keys():
                for target_seg_key in self.target_handle_name:
                    if target_seg_key in idToLabels[key]:
                        if self.target_handle_name in idToLabels[key]:
                            drawer_ids.append(int(key))

            valid_mask.append(
                torch.isin(pc[i, :, -1].reshape(-1),
                           torch.as_tensor(drawer_ids[i]).to(self.env.device)))

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

                # if index != self.segment_handle_camera_id:
                #     continue
                combined_mask[index] |= valid_handle_masks[
                    index]  # Accumulate all valid masks into one
                segmentation = torch.zeros((rgb.shape[1], rgb.shape[2], 3),
                                           dtype=torch.uint8,
                                           device=rgb.device)
                segmentation[combined_mask[index]] = rgb[index, ..., :3][
                    combined_mask[index]]
                # import cv2
                # cv2.imwrite("seg.png", segmentation.cpu().numpy())

                masks, boxes, labels, scores = self.sam_tool.get_masks(
                    (segmentation.cpu().numpy() / 255).astype(np.float32),
                    "handle.")

                # masks, boxes, labels, scores = self.sam_tool.get_masks((rgb[index, ..., :3].cpu().numpy()), "handle")
                overlay_region = []
                overlay_region_count = []
                for mask in masks:
                    handle_mask = torch.as_tensor(mask).to(self.env.device)
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

    def step_env(self,
                 last_obs,
                 target_traj,
                 collect_grasp=False,
                 collect_placement=False,
                 collect_cabinet=False,
                 collect_close=False):

        # Iterate through each step in the trajectory

        for i in range(len(target_traj)):
            # Perform the environment step
            observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                target_traj[i], collect_grasp, collect_placement,
                collect_cabinet, collect_close)

            if self.collector_interface is not None:
                # Iterate over buffer_map to check and update buffers
                for flag, buffers in step_buffer_map.items():
                    if locals(
                    )[flag]:  # Dynamically check if the corresponding flag is True
                        getattr(self, buffers[0]).append(
                            last_obs["policy"])  # Observation
                        getattr(self, buffers[1]).append(actions)  # Actions
                        getattr(self, buffers[2]).append(reward)  # Rewards
                        getattr(self, buffers[3]).append(
                            terminate)  # Termination status

            # Update last observation
            last_obs = observation

        return last_obs, success

    def replay_data(
        self,
        actions_buffer,
        init_open=False,
    ):
        observation = self.multi_env.step_reset(init_open, random_joint=False)

        for i in range(len(actions_buffer)):

            self.env.step(actions_buffer[i])

    def forward_manipulate_env(self,
                               target_manipulate_env=None,
                               reset_cabinet=False,
                               reset_grasp=False,
                               reset_close=False,
                               obs=None):
        success = False

        if self.use_demo_data:  # use demo data
            if self.segment_handle and self.handle_mask is None:

                for i in range(20):
                    self.env.sim.step()
            self.extract_data()
            self.env.reset()
            obs = self.multi_env.reset_demo_env(demo=self.demo)
            if self.multi_env.randomize_camera_pose:
                self.multi_env.random_camera_pose()

            if self.use_bounding_box:
                self.init_bbox = None
            if self.segment_handle:
                self.handle_points = self.sam_for_handle_segmentation(obs)
        else:
            if self.init_grasp and self.init_open and reset_grasp:
                print("connecting the policy")
                # target_manipulate_env.connect_policy(obs)

                # init_ee_pose = obs["policy"]["ee_pose"]

                # leave_cabinet_pose = init_ee_pose.clone()

                # for i in range(20):
                #     leave_cabinet_pose[:, -1] = 1
                #     leave_cabinet_pose[:, 2] -= 0.01
                #     if i > 10:
                #         leave_cabinet_pose[:, :
                #                            2] = target_manipulate_env.init_ee_pose[
                #                                0][:2].clone()

                #     observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                #         leave_cabinet_pose[0],
                #         collect_grasp=False,
                #         collect_placement=False,
                #         collect_cabinet=True,
                #         collect_close=False)

                target_manipulate_env.connect_policy(obs)

                connect_traj = target_manipulate_env.target_connect_traj

                for i in range(connect_traj.shape[0]):
                    observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                        connect_traj[i],
                        collect_grasp=True,
                        collect_placement=False,
                        collect_cabinet=False,
                        collect_close=False)

                target_manipulate_env.reset(observation)
            else:

                obs = self.multi_env.reset_all_env(reset_cabinet=reset_cabinet,
                                                   reset_grasp=reset_grasp,
                                                   reset_close=reset_close,
                                                   reset_all=self.reset_all)

            if obs is not None:
                obs, success = self.step_env(
                    obs,
                    target_manipulate_env.target_ee_traj,
                    collect_close=reset_close,
                    collect_grasp=reset_grasp,
                    collect_placement=False,
                    collect_cabinet=reset_cabinet)

        return obs, success

    def cache_data(self, ):
        # Map success flags to their corresponding interface strings and buffers

        stop_collect = False
        collect_cabinet, collect_close, collect_grasp, collect_placement = self.init_open, self.init_close, self.init_grasp, False

        # Iterate through the map and cache data if the success flag is True
        for flag, buffers in step_buffer_map.items():

            if locals()[flag]:  # Dynamically check the success flag
                # Dynamically get the interface attribute from collector_interface

                # Add demonstrations to buffer
                stop_collect = self.collector_interface.add_demonstraions_to_buffer(
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

    def step_multi_env(self):
        stop_collect = False
        observation = None

        #=========================================================================================================
        # open the cabinet
        #=========================================================================================================

        if self.init_open:
            observation, self.cabinet_success = self.forward_manipulate_env(
                self.multi_env.env_cabinet, reset_cabinet=True)
        #=========================================================================================================
        # grasp object
        #=========================================================================================================

        if self.init_grasp:
            observation, grasp_success = self.forward_manipulate_env(
                self.multi_env.env_grasp, reset_grasp=True, obs=observation)

            # place object happens after grasp object
            if self.init_placement and observation is not None:

                if self.placement_region is not None:

                    random_placement_pose = placement_region(
                        self.placement_region)
                else:
                    random_placement_pose = None

                result = self.multi_env.env_placement.get_target_placement_traj(
                    target_object_pose=random_placement_pose,
                    target_object_quat=None)
                if result is not None:
                    observation, self.grasp_success = self.step_env(
                        observation,
                        self.multi_env.env_placement.target_ee_traj,
                        collect_placement=True)
                    print("place object")

                else:
                    self.grasp_success = False

        if self.init_close:
            observation, self.close_success = self.forward_manipulate_env(
                self.multi_env.env_cabinetclose, reset_close=True)
        success = self.grasp_success or self.close_success or self.cabinet_success

        if self.collector_interface is not None:
            if success:
                stop_collect = self.cache_data()
            if self.grasp_success:
                self.reset_all = False
                self.grasp_success_count += 1

                if self.grasp_success_count == self.multi_env.success_pick_threhold:
                    self.reset_all = True
                    self.grasp_success_count = 0

            # reset success flag
            self.grasp_success = False
            self.close_success = False
            self.cabinet_success = False
            reset_data_buffer(self,
                              reset_close=True,
                              reset_cabinet=True,
                              reset_grasp=True,
                              reset_placement=True)
        return stop_collect

    #=========================================================================================================
    # replay data
    #=========================================================================================================

    def extract_data(self):

        self.demo = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]
        self.num_demos = len(self.collector_interface.raw_data["data"])

    def transform_handle(self, observation):
        self.sam_for_handle_segmentation(observation)

        handle_object = self.env.scene[self.target_handle_name]
        handle_root_state = handle_object.data.root_state_w
        trasformed_handle_vertices = math_utils.transform_points(
            self.handle_vertices.unsqueeze(0), handle_root_state[:, :3],
            handle_root_state[:, 3:7])

        indices = torch.randperm(
            trasformed_handle_vertices.shape[1])[:1024].to(self.env.device)
        seg_pc = observation["policy"]["seg_pc"][..., :3]
        # handle_pc = seg_pc.reshape(-1, 3)

        # o3d = vis_pc(trasformed_handle_vertices[0, :, :3].cpu().numpy(), None)
        # visualize_pcd([o3d])

        sampled_pc = trasformed_handle_vertices[:, indices].unsqueeze(0)
        observation["policy"]["seg_pc"] = torch.cat([seg_pc, sampled_pc],
                                                    dim=2)
        return observation

    def step_unnormalized_env(self,
                              skip_frame=1,
                              start_frame=0,
                              reset_every_step=True,
                              succes_bool=True):
        last_obs, _ = self.forward_manipulate_env()

        collect_cabinet, collect_close, collect_grasp, collect_placement = self.init_open, self.init_close, self.init_grasp, False

        if not self.use_joint_pos:
            action_normailized_range = 3

            raw_actions = torch.as_tensor(np.array(self.demo["actions"]))
            raw_actions = raw_actions.reshape(-1, raw_actions.shape[-1])

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

        normalized_action = normalized_action[start_frame::skip_frame]

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

            else:
                unnormalized_action = unnormalized_action[..., :8]

            observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                unnormalized_action,
                collect_grasp=self.init_grasp and not self.init_placement,
                collect_placement=self.init_placement,
                collect_cabinet=self.init_open,
                collect_close=self.init_close)

            # if self.seg_handle:
            #     observation = self.transform_handle(observation)

            #     # Create a sample point cloud (or load your own .ply/.pcd file)
            # seg_pc = observation["policy"]["seg_pc"]
            # seg_pc = seg_pc.reshape(-1, seg_pc.shape[-1]).cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(
            #     seg_pc[:, :3])  # 1000 random points

            # # Visualize the point cloud
            # o3d.visualization.draw_geometries([pcd])

            for flag, buffers in step_buffer_map.items():
                if locals(
                )[flag]:  # Dynamically check if the corresponding flag is True
                    getattr(self, buffers[0]).append(
                        last_obs["policy"])  # Observation

                    getattr(self, buffers[1]).append(
                        normalized_action[i].unsqueeze(0))  # Actions
                    getattr(self, buffers[2]).append(reward)  # Rewards
                    getattr(self,
                            buffers[3]).append(terminate)  # Termination status

                    presave_buffer = [
                        getattr(self, buffers[0]),
                        getattr(self, buffers[1]),
                        getattr(self, buffers[2]),
                        getattr(self, buffers[3])
                    ]

            last_obs = observation
            # last_obs = self.process_pc(last_obs)

            # seg_pc = last_obs["policy"]["seg_pc"][0].cpu().numpy()
            # o3d_pc = vis_pc(seg_pc[:, :3], seg_pc[:, 3:6])
            # visualize_pcd([o3d_pc])
        stop = False

        if reset_every_step:  # do not save the failure data
            if success == succes_bool:

                self.cache_data()

            reset_data_buffer(self,
                              reset_grasp=True,
                              reset_cabinet=True,
                              reset_close=True,
                              reset_placement=True)

            self.demo_index += 1  # move to next episode

            return stop
        else:
            return success, copy.deepcopy(presave_buffer)

    def replay_normalized_action(self, ):
        last_obs, _ = self.forward_manipulate_env()

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
                    self.multi_env.device))
        self.demo_index += 1
