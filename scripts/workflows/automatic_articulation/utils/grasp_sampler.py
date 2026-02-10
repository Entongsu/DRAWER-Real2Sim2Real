import sys

sys.path.append("../M2T2")

import hydra
import numpy as np
import torch

from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (create_visualizer, make_frame, visualize_grasp,
                                visualize_pointcloud)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu

from omegaconf import OmegaConf
import sys
import pymeshlab

sys.path.append(".")
from curobo.util.usd_helper import UsdHelper
from tools.visualization_utils import *
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from curobo.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    Obstacle,
    Sphere,
    WorldConfig,
)
from curobo.types.math import Pose
from tools.curobo_planner import get_mesh_attrs
from pxr import Usd, UsdGeom


class GraspSampler:

    def __init__(self,
                 env,
                 cfg,
                 checkpoint_path,
                 synthesis_points=True,
                 env_config=None):

        self.cfg = OmegaConf.load(cfg)

        self.checkpoint_path = checkpoint_path
        self.env = env
        self.env_config = env_config

        # load the model
        self.model = M2T2.from_config(self.cfg.m2t2)
        ckpt = torch.load(self.checkpoint_path)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.cuda().eval()

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(env.scene.stage)

        self.synthesis_points = synthesis_points
        if "kitchen" in self.env.scene.keys():

            self.grasp_object = self.env.scene[self.env_config["params"]
                                               ["Task"]["target_object"]]
            self.grasp_object_name = self.env_config["params"]["Task"][
                "target_object"]
        else:
            self.grasp_object = None
            self.grasp_object_name = None

    def filter_out_lables(self, seg_pc, labels):

        background_mask = seg_pc[..., -1] == labels
        fg_mask = ~background_mask
        seg_pc = seg_pc[fg_mask]
        return seg_pc

    def get_target_prim(self):

        scale = None
        translate = None
        euler_angle = None

        geometries = []
        all_items = self.usd_help.stage.Traverse()

        # Get the prim that contains the target object
        prim = [
            prim for prim in all_items if prim.IsA(UsdGeom.Mesh)
            and self.grasp_object_name in prim.GetPath().pathString
        ][0]

        if (prim.GetAttribute("points").Get() is not None):
            points = torch.as_tensor(prim.GetAttribute("points").Get())
            if prim.GetAttribute("xformOp:scale").Get() is not None:
                scale = torch.as_tensor(
                    prim.GetAttribute("xformOp:scale").Get())

            if prim.GetParent().GetAttribute(
                    "xformOp:translate").Get() is not None:

                translate = torch.as_tensor(
                    prim.GetParent().GetAttribute("xformOp:translate").Get())
            if prim.GetParent().GetAttribute(
                    "xformOp:orientXYZ").Get() is not None:
                euler_angle = torch.as_tensor(
                    prim.GetParent().GetAttribute("xformOp:orientXYZ").Get())

            faces = list(prim.GetAttribute("faceVertexIndices").Get())
            faces = np.array(faces).reshape(-1, 3)

            geometries.append(points)

        if translate is None:
            translate = torch.zeros(3)
        if euler_angle is None:
            euler_angle = torch.zeros(3)
        # if scale is None:
        # TODO: Need to revisit this for the crrectness
        if self.grasp_object is not None:
            scale = torch.ones(3) * self.grasp_object.cfg.rigid_cfg["scale"][0]
        else:
            scale = torch.ones(3)

        quaternion = math_utils.quat_from_euler_xyz(euler_angle[0],
                                                    euler_angle[1],
                                                    euler_angle[2])
        return geometries, faces, scale, translate, quaternion

    def sythesis_points(self):

        robot_root_state = self.env.scene["robot"]._data.root_state_w[0, :7]
        geometries, faces, scale, translate, quaternion = self.get_target_prim(
        )

        # transform the object to the robot frame
        geometries = torch.cat(geometries, dim=0)
        vertices = (scale * math_utils.transform_points(
            geometries, translate, quaternion)).to("cuda:0")

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
            vertices = torch.tensor(vertices, dtype=torch.float32).to("cuda:0")

        if self.grasp_object is not None:
            grasp_object_pose = self.grasp_object._data.root_state_w[:, :7]
        else:
            true_prim_name = '_'.join(
                self.grasp_object_name.lower().split('_')[:2])
            rigid_collections = self.env.scene[true_prim_name]
            rigid_bodies_id, _ = rigid_collections.find_objects(
                self.grasp_object_name)
            grasp_object_pose = rigid_collections._data.object_state_w[:,
                                                                       rigid_bodies_id[
                                                                           0], :
                                                                       7]
        robot_root_state = self.env.scene["robot"]._data.root_state_w[:, :7]
        self.robot2object_poe, self.robot2object_quat = math_utils.subtract_frame_transforms(
            robot_root_state[:, :3], robot_root_state[:, 3:7],
            grasp_object_pose[:, :3], grasp_object_pose[:, 3:7])
        rotated_vertice = math_utils.transform_points(
            vertices, self.robot2object_poe[0], self.robot2object_quat[0])
        aug_seg = torch.ones(rotated_vertice.shape[0]).to("cuda:0")
        # aug_seg[..., 0] = 255
        aug_input = torch.cat(
            [rotated_vertice,
             torch.zeros_like(rotated_vertice) + 255], dim=1)
        aug_input[:, :3] -= torch.mean(aug_input[:, :3], dim=0)
        aug_points = rotated_vertice.clone()

        return aug_seg, aug_input, aug_points

    def load_data(self, obs):

        data = {}

        # if "seg_pc" in obs.keys():
        #     continue

        #     seg_pc = obs["seg_pc"]

        #     for key in self.env.scene.keys():
        #         if "camera" in key:
        #             cam = key
        #             break

        #     idToLabels = self.env.scene[cam].data.info[0][
        #         "instance_segmentation_fast"]["idToLabels"]

        #     seg_pc = seg_pc.reshape(-1, 7)

        #     for key in idToLabels.keys():

        #         label = int(key)
        #         if self.env.scene["kitchen"].cfg.articulation_cfg[
        #                 "target_object"] not in idToLabels[key]:
        #             seg_pc = self.filter_out_lables(seg_pc, label)
        #     if len(seg_pc) > 0:

        #         data["seg"] = seg_pc[:, -1]
        #         data["inputs"] = seg_pc[:, 0:6].clone()
        #         data["inputs"][:, :3] -= torch.mean(data["inputs"][:, :3],
        #                                             dim=0)
        #         data["points"] = seg_pc[:, :3].clone()
        #     else:
        #         data["seg"] = []
        #         data["inputs"] = []
        #         data["points"] = []

        data['task'] = 'pick'

        data['ee_pose'] = torch.eye(4)
        data['bottom_center'] = torch.zeros(3)
        data['object_center'] = torch.zeros(3)

        if self.synthesis_points:
            aug_seg, aug_input, aug_points = self.sythesis_points()

            # if "seg_pc" in obs.keys():
            #     data['seg'] = torch.cat([data['seg'], aug_seg], dim=0)
            #     data["inputs"] = torch.cat([data["inputs"], aug_input], dim=0)
            #     data["points"] = torch.cat([data["points"], aug_points], dim=0)
            #     data['object_inputs'] = data["inputs"].clone()
            # else:
            data['seg'] = aug_seg
            data["inputs"] = aug_input
            data["points"] = aug_points
            data['object_inputs'] = data["inputs"].clone()

        return data

    def load_and_predict(self, obs):
        data = self.load_data(obs)

        inputs, xyz, seg = data['inputs'], data['points'], data['seg']
        obj_inputs = data['object_inputs']

        outputs = {
            'grasps': [],
            'grasp_confidence': [],
            'grasp_contacts': [],
            'placements': [],
            'placement_confidence': [],
            'placement_contacts': []
        }
        for _ in range(self.cfg.eval.num_runs):
            pt_idx = sample_points(xyz, self.cfg.data.num_points)
            data['inputs'] = inputs[pt_idx]
            data['points'] = xyz[pt_idx]
            data['seg'] = seg[pt_idx]
            # pt_idx = sample_points(obj_inputs, self.cfg.data.num_object_points)
            data['object_inputs'] = obj_inputs[pt_idx]
            data_batch = collate([data])
            to_gpu(data_batch)

            with torch.no_grad():
                model_ouputs = self.model.infer(data_batch, self.cfg.eval)

            for key in outputs:
                if 'place' in key and len(outputs[key]) > 0:
                    outputs[key] = [
                        torch.cat([prev, cur]) for prev, cur in zip(
                            outputs[key], model_ouputs[key][0])
                    ]
                else:
                    outputs[key].extend(model_ouputs[key][0])

        # data['inputs'], data['points'], data['seg'] = inputs.cpu(), xyz.cpu(
        # ), seg.cpu()
        data['object_inputs'] = obj_inputs
        return data, outputs

    def visualization_select_pose(self, data, outputs, grasp):
        to_cpu(outputs)
        to_cpu(data)
        vis = create_visualizer()
        rgb = denormalize_rgb(data['inputs'][:,
                                             3:].T.unsqueeze(2)).squeeze(2).T
        rgb = (rgb.numpy()).astype('uint8')

        xyz = data['points'].numpy()

        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)

        for index, gs in enumerate(grasp[0]):

            visualize_grasp(vis,
                            f"object_{index:02d}/grasps/{index:03d}",
                            gs.cpu().numpy(),
                            get_set_colors()[0],
                            linewidth=2)

    def visualization(self, data, outputs):
        to_cpu(outputs)
        to_cpu(data)

        vis = create_visualizer()
        rgb = denormalize_rgb(data['inputs'][:,
                                             3:].T.unsqueeze(2)).squeeze(2).T
        rgb = (rgb.numpy()).astype('uint8')

        xyz = data['points'].numpy()
        # cam_pose = data['cam_pose'].double().numpy()
        # make_frame(vis, 'camera', T=cam_pose)
        if not self.cfg.eval.world_coord:
            xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)

        if data['task'] == 'pick':
            for i, (grasps, conf, contacts, color) in enumerate(
                    zip(outputs['grasps'], outputs['grasp_confidence'],
                        outputs['grasp_contacts'], get_set_colors())):
                print(f"object_{i:02d} has {grasps.shape[0]} grasps")
                conf = conf.numpy()
                conf_colors = (np.stack(
                    [1 - conf, conf, np.zeros_like(conf)], axis=1) *
                               255).astype('uint8')
                visualize_pointcloud(vis,
                                     f"object_{i:02d}/contacts",
                                     contacts.numpy(),
                                     conf_colors,
                                     size=0.01)
                grasps = grasps.numpy()
                if not self.cfg.eval.world_coord:
                    grasps = cam_pose @ grasps
                for j, grasp in enumerate(grasps):
                    visualize_grasp(vis,
                                    f"object_{i:02d}/grasps/{j:03d}",
                                    grasp,
                                    color,
                                    linewidth=0.2)
        elif data['task'] == 'place':
            ee_pose = data['ee_pose'].double().numpy()
            make_frame(vis, 'ee', T=ee_pose)
            obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
            obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
            obj_xyz = obj_xyz_ee @ ee_pose[:3, :3].T + ee_pose[:3, 3]
            obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
            obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
            visualize_pointcloud(vis, 'object', obj_xyz, obj_rgb, size=0.005)
            for i, (placements, conf, contacts) in enumerate(
                    zip(
                        outputs['placements'],
                        outputs['placement_confidence'],
                        outputs['placement_contacts'],
                    )):
                print(
                    f"orientation_{i:02d} has {placements.shape[0]} placements"
                )
                conf = conf.numpy()
                conf_colors = (np.stack(
                    [1 - conf, conf, np.zeros_like(conf)], axis=1) *
                               255).astype('uint8')
                visualize_pointcloud(vis,
                                     f"orientation_{i:02d}/contacts",
                                     contacts.numpy(),
                                     conf_colors,
                                     size=0.01)
                placements = placements.numpy()
                if not self.cfg.eval.world_coord:
                    placements = cam_pose @ placements
                visited = np.zeros((0, 3))
                for j, k in enumerate(
                        np.random.permutation(placements.shape[0])):
                    if visited.shape[0] > 0:
                        dist = np.sqrt(
                            ((placements[k, :3, 3] - visited)**2).sum(axis=1))
                        if dist.min() < self.cfg.eval.placement_vis_radius:
                            continue
                    visited = np.concatenate(
                        [visited, placements[k:k + 1, :3, 3]])
                    visualize_grasp(
                        vis,
                        f"orientation_{i:02d}/placements/{j:02d}/gripper",
                        placements[k], [0, 255, 0],
                        linewidth=0.2)
                    obj_xyz_placed = obj_xyz_ee @ placements[k, :3, :3].T \
                                + placements[k, :3, 3]
                    visualize_pointcloud(
                        vis,
                        f"orientation_{i:02d}/placements/{j:02d}/object",
                        obj_xyz_placed,
                        obj_rgb,
                        size=0.01)
