import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
import os
import pickle

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux
import transformations

from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, UsdUtils, Sdf, PhysxSchema
from load_obj_utils import load_obj
import isaaclab.utils.math as math_utils
import torch

global_sdf = True


def create_rigid_collision(prim, set_sdf=True, sdf_resolution=512):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
    if set_sdf and global_sdf:
        meshcollisionAPI.CreateApproximationAttr().Set("sdf")
        meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        meshCollision.CreateSdfResolutionAttr().Set(sdf_resolution)
    else:
        meshcollisionAPI.CreateApproximationAttr().Set("convexDecomposition")


def add_fixed(stage, joint_path, parent_path, child_path, rel0, relrot0):
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    parent_prim = stage.GetPrimAtPath(parent_path)
    child_prim = stage.GetPrimAtPath(child_path)
    fixed_joint.GetBody0Rel().SetTargets([parent_prim.GetPath()])
    fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])

    # define revolute joint local poses for bodies
    fixed_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))

    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    fixed_joint.CreateLocalRot0Attr().Set(quat)

    return stage


def add_revolute(stage, joint_path, main_path, door_path, rel0, relrot0, rel1,
                 inverse):
    jointPath = joint_path
    revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, jointPath)

    # define revolute joint bodies
    revoluteJoint.CreateBody0Rel().SetTargets([main_path])
    revoluteJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    revoluteJoint.CreateAxisAttr("Z")
    if inverse:
        revoluteJoint.CreateLowerLimitAttr(-90.0)
        revoluteJoint.CreateUpperLimitAttr(0.0)
    else:
        revoluteJoint.CreateLowerLimitAttr(0.0)
        revoluteJoint.CreateUpperLimitAttr(90.0)

    # define revolute joint local poses for bodies
    revoluteJoint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))

    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    revoluteJoint.CreateLocalRot0Attr().Set(quat)

    revoluteJoint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    revoluteJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    revoluteJoint.CreateBreakForceAttr().Set(1e20)
    revoluteJoint.CreateBreakTorqueAttr().Set(1e20)

    return stage


def add_prismatic(stage, joint_path, main_path, door_path, rel0, relrot0,
                  rel1):
    jointPath = joint_path
    prismaticJoint = UsdPhysics.PrismaticJoint.Define(stage, jointPath)

    # define revolute joint bodies
    prismaticJoint.CreateBody0Rel().SetTargets([main_path])
    prismaticJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    prismaticJoint.CreateAxisAttr("X")
    prismaticJoint.CreateLowerLimitAttr(0.0)
    prismaticJoint.CreateUpperLimitAttr(0.2 * scale_factor)

    # define revolute joint local poses for bodies
    prismaticJoint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))
    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    prismaticJoint.CreateLocalRot0Attr().Set(quat)

    prismaticJoint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    prismaticJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    prismaticJoint.CreateBreakForceAttr().Set(1e20)
    prismaticJoint.CreateBreakTorqueAttr().Set(1e20)

    prismaticDriveAPI = UsdPhysics.DriveAPI.Apply(
        stage.GetPrimAtPath(jointPath), "linear")
    prismaticDriveAPI.CreateTypeAttr("force")
    prismaticDriveAPI.CreateMaxForceAttr(1e20)
    prismaticDriveAPI.CreateTargetVelocityAttr(0.5)
    prismaticDriveAPI.CreateDampingAttr(1e10)
    prismaticDriveAPI.CreateStiffnessAttr(0.0)

    return stage


def parse_urdf_with_origin_and_collision(urdf_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Initialize the dictionary
    urdf_data = {
        "links": {},  # Dictionary to store link information
        "joints": {},  # Dictionary to store joint information
    }

    # Parse links
    for link in root.findall("link"):
        link_name = link.get("name")

        # Parse visual meshes
        visual_meshes = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            origin = visual.find("origin")
            name = visual.get("name")
            visual_origin = {
                "xyz": origin.get("xyz") if origin is not None else None,
                "rpy": origin.get("rpy") if origin is not None else None,
            }
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    visual_meshes.append({
                        "filename": mesh.get("filename"),
                        "origin": visual_origin,
                        "name": name,
                    })

        # Parse collision meshes
        collision_meshes = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            origin = collision.find("origin")
            collision_origin = {
                "xyz": origin.get("xyz") if origin is not None else None,
                "rpy": origin.get("rpy") if origin is not None else None,
            }
            name = visual.get("name")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    collision_meshes.append({
                        "filename": mesh.get("filename"),
                        "origin": collision_origin,
                        "name": name,
                    })

        urdf_data["links"][link_name] = {
            "visual_meshes": visual_meshes,
            "collision_meshes": collision_meshes,
        }

    # Parse joints
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")

        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")

        joint_origin = {
            "xyz": origin.get("xyz") if origin is not None else None,
            "rpy": origin.get("rpy") if origin is not None else None,
        }

        urdf_data["joints"][joint_name] = {
            "type": joint_type,
            "parent": parent,
            "child": child,
            "origin": joint_origin,
        }

    return urdf_data


def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)


def AddRotate(top, quat):
    top.AddOrientOp().Set(value=Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))


def AddScale(top, scale):
    top.AddScaleOp().Set(value=scale)


def add_fixed(stage, joint_path, parent_path, child_path, rel0, relrot0):
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    parent_prim = stage.GetPrimAtPath(parent_path)
    child_prim = stage.GetPrimAtPath(child_path)
    fixed_joint.GetBody0Rel().SetTargets([parent_prim.GetPath()])
    fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])

    # define revolute joint local poses for bodies
    fixed_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))

    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    fixed_joint.CreateLocalRot0Attr().Set(quat)

    return stage


def create_xform(stage, scale, translation, rotation, path):
    xform = UsdGeom.Xform.Define(stage, path)

    AddTranslate(xform, translation)
    AddRotate(xform, rotation)
    AddScale(xform, (scale[0], scale[1], scale[2]))

    return stage


class AssembleArticulation:

    def __init__(self,
                 ckpt_dir,
                 root='/kitchen',
                 usda_name='kitchen.usda',
                 urdf_dict=None):
        # Path to config YAML file.
        # ckpt_dir = "./scannetpp/2e67a32314/"
        self.ckpt_dir = ckpt_dir  #"/media/aurmr/data1/weird/IsaacLab/tools/auto_articulation/asset"
        self.usda_name = usda_name
        self.urdf_dict = urdf_dict
        self.root = '/kitchen'
        self.scale = (1, 1, 1)

        self.set_env()

    def set_env(self):

        self.stage = Usd.Stage.CreateNew(
            os.path.join(self.ckpt_dir, self.usda_name))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        modelRoot = UsdGeom.Xform.Define(self.stage, self.root)
        Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

        light = UsdLux.DomeLight.Define(self.stage, f"{self.root}/light")

        # Set DomeLight-specific attributes
        light.CreateExposureAttr(3.0)
        light.CreateIntensityAttr(100.0)
        light.CreateColorAttr((1.0, 1.0, 1.0))

        stage = create_xform(
            self.stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}")
        stage = create_xform(
            stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}/sektion")
        prim = stage.GetPrimAtPath(f"{self.root}")
        stage.SetDefaultPrim(prim)

    def convert(self):
        self.assembly_link()
        self.assembly_joint()

        # fixed_joint = UsdPhysics.FixedJoint.Define(self.stage,
        #                                            self.root + "/fixed_joint")

        # child_prim = self.stage.GetPrimAtPath(self.root + "/sektion")
        # fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])
        # self.stage = add_fixed(self.stage,
        #                        f"{self.root}/main_frame/fixed_joint",
        #                        f"{self.root}/sektion",
        #                        f"{self.root}/main_frame", (0.0, 0.0, 0.0),
        #                        (1, 0, 0, 0))

        prim = self.stage.GetPrimAtPath(self.root)
        articulation_root = UsdPhysics.ArticulationRootAPI.Apply(prim)
        self.stage.Save()

    def assembly_joint(self):
        joints_dict = self.urdf_dict["joints"]
        for joint_info in joints_dict.keys():
            joint_dict = joints_dict[joint_info]
            if joint_dict["type"] == "fixed":

                parent_path = self.root + '/' + joint_dict["parent"]
                child_path = self.root + '/' + joint_dict["child"]
                rel0, relrot0 = self.get_xyz_rpy(joint_dict["origin"]["xyz"],
                                                 joint_dict["origin"]["rpy"])
                add_fixed(self.stage,
                          self.root + f'/{joint_dict["parent"]}/' + joint_info,
                          parent_path, child_path, rel0, relrot0)

    def get_xyz_rpy(self, xyz_string, rpy_string):

        xyz_array = np.array([float(value) for value in xyz_string.split()])

        if rpy_string is None:
            rpy_array = np.array([0.0, 0.0, 0.0])
        else:
            rpy_array = np.array(
                [float(value) for value in rpy_string.split()])
        rotation = transformations.quaternion_from_euler(rpy_array[0],
                                                         rpy_array[1],
                                                         rpy_array[2],
                                                         axes='sxyz')
        return xyz_array, rotation

    def assembly_meshes(self, link_info, target_meshes, mesh_type="visuals"):

        for index, target_mesh in enumerate(target_meshes):
            prim_mesh = UsdGeom.Xform.Define(
                self.stage, self.root + '/' + link_info + f"/{mesh_type}")
            if target_mesh["name"] is not None:
                prim_mesh = UsdGeom.Mesh.Define(
                    self.stage,
                    self.root + '/' + link_info + f"/{mesh_type}/" +
                    target_mesh["name"].replace("-", "_") + "_" + str(index))
            else:
                prim_mesh = UsdGeom.Mesh.Define(
                    self.stage, self.root + '/' + link_info +
                    f"/{mesh_type}/mesh_" + str(index))

            mesh_path = os.path.join(self.ckpt_dir, target_mesh["filename"])
            raw_mesh = trimesh.load(mesh_path)

            if isinstance(raw_mesh, trimesh.Scene):
                # Extract all meshes in the scene
                meshes = [geom for geom in raw_mesh.geometry.values()]

                # Merge all meshes into a single mesh
                combined_mesh = trimesh.util.concatenate(meshes)

                # Access points and faces
                points = combined_mesh.vertices
                faces = combined_mesh.faces
            else:
                points = raw_mesh.vertices
                faces = raw_mesh.faces

            vertex_counts = np.ones_like(faces[:, 0]).reshape(-1).astype(
                np.int32) * 3

            prim_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            prim_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
            prim_mesh.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(vertex_counts))
            xyz_array, rotation = self.get_xyz_rpy(
                target_mesh["origin"]["xyz"], target_mesh["origin"]["rpy"])

            AddRotate(prim_mesh, rotation)
            AddTranslate(prim_mesh, (xyz_array[0], xyz_array[1], xyz_array[2]))

            if mesh_type == "collisions":

                collision_prim = self.stage.GetPrimAtPath(
                    prim_mesh.GetPath().pathString)

                create_rigid_collision(collision_prim)

    def assembly_link(self):

        links_dict = self.urdf_dict["links"]
        for link_info in links_dict.keys():
            link_dict = links_dict[link_info]
            create_xform(self.stage, self.scale, (0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0, 1.0), self.root + '/' + link_info)
            visual_meshes = link_dict["visual_meshes"]

            collision_meshes = link_dict["collision_meshes"]
            if len(visual_meshes) == 0:
                continue
            self.assembly_meshes(link_info, visual_meshes, mesh_type="visuals")
            if len(collision_meshes) > 0:

                self.assembly_meshes(link_info,
                                     collision_meshes,
                                     mesh_type="collisions")


# Example usage
urdf_file_path = "source/assets/kitchen/usd/cabinet02/raw_data/modified_mobility.urdf"

# Parse URDF file into a dictionary
urdf_dict = parse_urdf_with_origin_and_collision(urdf_file_path)

assembly = AssembleArticulation(
    "source/assets/kitchen/usd/cabinet02/raw_data/",
    usda_name='modified_mobility.usda',
    urdf_dict=urdf_dict)
assembly.convert()
