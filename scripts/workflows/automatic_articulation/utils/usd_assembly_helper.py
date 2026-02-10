import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
import os
import pickle

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux
import transformations

from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, UsdUtils, Sdf, PhysxSchema
from load_obj_utils import load_obj
import isaaclab.utils.math as math_utils
import torch

global_sdf = True


def parse_mtl(mtl_path):
    materials = {}
    current_mtl = None

    with open(mtl_path, 'r') as mtl_file:
        for line in mtl_file:
            if line.startswith('newmtl'):
                current_mtl = line.split()[1]
                materials[current_mtl] = {}
            elif current_mtl:
                parts = line.split()
                if parts:
                    key = parts[0]
                    value = parts[1:]
                    materials[current_mtl][key] = value

    return materials


def create_usd_material(stage, material_path, material_name, material_data,
                        texture_dir):
    # Define the Material
    material = UsdShade.Material.Define(stage, material_path)

    # Create the UsdPreviewSurface Shader
    pbrShader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
    pbrShader.CreateIdAttr("UsdPreviewSurface")

    # Set roughness and metallic inputs
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Connect the shader to the material's surface output
    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(),
                                                   "surface")

    # Create a Primvar Reader for texture coordinates
    stReader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    stReader.CreateIdAttr('UsdPrimvarReader_float2')

    # Create an input on the material for the primvar name
    stInput = material.CreateInput('frame:stPrimvarName',
                                   Sdf.ValueTypeNames.Token)
    stInput.Set('st')

    # Connect the stReader's varname input to the material's stInput
    stReader.CreateInput('varname',
                         Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

    # Handle diffuse color or texture
    if 'Kd' in material_data:
        diffuse_color = [float(c) for c in material_data['Kd']]
        pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*diffuse_color))

    if 'map_Kd' in material_data:
        texture_file = os.path.join(texture_dir, material_data['map_Kd'][0])

        # Create a UsdUVTexture shader for the diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(
            stage, f"{material_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput(
            'file', Sdf.ValueTypeNames.Asset).Set(texture_file)

        # Connect the texture sampler to the stReader for texture coordinates
        diffuseTextureSampler.CreateInput(
            "st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)

        # Connect the texture sampler's output to the pbrShader's diffuseColor input
        pbrShader.CreateInput("diffuseColor",
                              Sdf.ValueTypeNames.Color3f).ConnectToSource(
                                  diffuseTextureSampler.ConnectableAPI(),
                                  'rgb')

    return material


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


def determine_axis(axis_str):
    # Convert the axis string into a NumPy array
    axis = np.array([float(v) for v in axis_str.split()])

    # Normalize the vector to handle arbitrary magnitudes
    axis = axis / np.linalg.norm(axis)

    # Threshold for determining alignment
    threshold = 1e-6

    # Check which axis the vector aligns with
    if abs(abs(axis[0]) - 1) < threshold:
        return "X", axis[0]
    elif abs(abs(axis[1]) - 1) < threshold:
        return "Y", axis[1]
    elif abs(abs(axis[2]) - 1) < threshold:
        return "Z", axis[2]
    else:
        return "None (diagonal or arbitrary orientation)"


def add_revolute(stage, joint_path, main_path, door_path, rel0, relrot0, rel1,
                 axis, joint_limits):
    jointPath = joint_path
    revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, jointPath)

    # define revolute joint bodies
    revoluteJoint.CreateBody0Rel().SetTargets([main_path])
    revoluteJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees

    axis, axis_dir = determine_axis(axis)
    revoluteJoint.CreateAxisAttr(axis)

    if axis_dir < 0:

        joint_limits = [-joint_limits[1], -joint_limits[0]]

    revoluteJoint.CreateLowerLimitAttr(
        np.max([joint_limits[0] / 3.14 * 180, -90]))
    revoluteJoint.CreateUpperLimitAttr(
        np.min([joint_limits[1] / 3.14 * 180, 90]))

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
                  limits, axis):
    jointPath = joint_path
    prismaticJoint = UsdPhysics.PrismaticJoint.Define(stage, jointPath)

    # define revolute joint bodies
    prismaticJoint.CreateBody0Rel().SetTargets([main_path])
    prismaticJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    axis_dir, _ = determine_axis(axis)
    prismaticJoint.CreateAxisAttr(axis_dir)
    prismaticJoint.CreateLowerLimitAttr(limits[0])
    prismaticJoint.CreateUpperLimitAttr(limits[1])

    # define revolute joint local poses for bodies
    prismaticJoint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))
    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    prismaticJoint.CreateLocalRot0Attr().Set(quat)

    # prismaticJoint.CreateLocalPos1Attr().Set(
    #     Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    # prismaticJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

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
        limit = joint.find("limit")

        joint_origin = {
            "xyz": origin.get("xyz") if origin is not None else None,
            "rpy": origin.get("rpy") if origin is not None else None,
        }
        if limit is not None:
            joint_limit = {
                "lower":
                float(limit.get("lower"))
                if limit.get("lower") is not None else None,
                "upper":
                float(limit.get("upper"))
                if limit.get("upper") is not None else None,
                "effort":
                float(limit.get("effort"))
                if limit.get("effort") is not None else None,
                "velocity":
                float(limit.get("velocity"))
                if limit.get("velocity") is not None else None,
                "axis":
                joint.find("axis").get("xyz")
                if joint.find("axis") is not None else None,
            }

        else:
            joint_limit = None
        urdf_data["joints"][joint_name] = {
            "type": joint_type,
            "parent": parent,
            "child": child,
            "origin": joint_origin,
            "limit": joint_limit
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
