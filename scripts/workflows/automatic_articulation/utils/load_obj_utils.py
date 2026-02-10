import os

from iopath.common.file_io import PathManager
from typing import cast, ContextManager, IO, Optional, Union
import contextlib
import pathlib

import pathlib
import os
import copy
import isaacsim.core.utils.prims as prim_utils
from curobo.util.usd_helper import UsdHelper
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import trimesh
import numpy as np


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


def _parse_face(
    line,
    tokens,
    material_idx,
    faces_verts_idx,
    faces_normals_idx,
    faces_textures_idx,
    faces_materials_idx,
) -> None:
    face = tokens[1:]
    face_list = [f.split("/") for f in face]
    face_verts = []
    face_normals = []
    face_textures = []

    for vert_props in face_list:
        # Vertex index.

        face_verts.append(int(vert_props[0]))
        if len(vert_props) > 1:
            if vert_props[1] != "":
                # Texture index is present e.g. f 4/1/1.
                face_textures.append(int(vert_props[1]))
            if len(vert_props) > 2:
                # Normal index present e.g. 4/1/1 or 4//1.
                face_normals.append(int(vert_props[2]))
            if len(vert_props) > 3:
                raise ValueError("Face vertices can only have 3 properties. \
                                Face vert %s, Line: %s" %
                                 (str(vert_props), str(line)))

    # Triplets must be consistent for all vertices in a face e.g.
    # legal statement: f 4/1/1 3/2/1 2/1/1.
    # illegal statement: f 4/1/1 3//1 2//1.
    # If the face does not have normals or textures indices
    # fill with pad value = -1. This will ensure that
    # all the face index tensors will have F values where
    # F is the number of faces.
    if len(face_normals) > 0:
        if not (len(face_verts) == len(face_normals)):
            raise ValueError("Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s" %
                             (str(face), str(line)))
    else:
        face_normals = [-1] * len(face_verts)  # Fill with -1
    if len(face_textures) > 0:
        if not (len(face_verts) == len(face_textures)):
            raise ValueError("Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s" %
                             (str(face), str(line)))
    else:
        face_textures = [-1] * len(face_verts)  # Fill with -1

    # Subdivide faces with more than 3 vertices.
    # See comments of the load_obj function for more details.
    for i in range(len(face_verts) - 2):
        faces_verts_idx.append(
            (face_verts[0], face_verts[i + 1], face_verts[i + 2]))
        faces_normals_idx.append(
            (face_normals[0], face_normals[i + 1], face_normals[i + 2]))
        faces_textures_idx.append(
            (face_textures[0], face_textures[i + 1], face_textures[i + 2]))
        faces_materials_idx.append(material_idx)


def _parse_obj(f, data_dir: str):
    """
    Load a mesh from a file-like object. See load_obj function for more details
    about the return values.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None

    lines = [line.strip() for line in f]

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    materials_idx = -1

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: only allow one .mtl file per .obj.
            # Definitions for multiple materials can be included
            # in this one .mtl file.
            mtl_path = line[len(tokens[0]):].strip(
            )  # Take the remainder of the line
            mtl_path = os.path.join(data_dir, mtl_path)
        elif len(tokens) and tokens[0] == "usemtl":
            material_name = tokens[1]
            # materials are often repeated for different parts
            # of a mesh.
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        elif line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" %
                    (str(tx), str(line)))
            verts_uvs.append(tx)
        elif line.startswith("vn "):  # Line is a normal.
            norm = [float(x) for x in tokens[1:4]]
            if len(norm) != 3:
                msg = "Normal %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(norm), str(line)))
            normals.append(norm)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    # return (
    #     verts,
    #     normals,
    #     verts_uvs,
    #     faces_verts_idx,
    #     faces_normals_idx,
    #     faces_textures_idx,
    #     faces_materials_idx,
    #     material_names,
    #     mtl_path,
    # )

    return {
        "verts": verts,
        "normals": normals,
        "verts_uvs": verts_uvs,
        "faces_verts_idx": faces_verts_idx,
        "faces_normals_idx": faces_normals_idx,
        "faces_textures_idx": faces_textures_idx,
        "faces_materials_idx": faces_materials_idx,
        "material_names": material_names,
        "mtl_path": mtl_path,
    }


def _open_file(f,
               path_manager: PathManager,
               mode: str = "r") -> ContextManager[IO]:
    if isinstance(f, str):
        # pyre-fixme[6]: For 2nd argument expected `Union[typing_extensions.Literal['...
        f = path_manager.open(f, mode)
        return contextlib.closing(f)
    elif isinstance(f, pathlib.Path):
        f = f.open(mode)
        return contextlib.closing(f)
    else:
        return contextlib.nullcontext(cast(IO, f))


def load_obj(obj_path):
    data_dir = os.path.dirname(obj_path)
    path_manager = PathManager()

    with _open_file(obj_path, path_manager, "r") as f_obj:
        return _parse_obj(f_obj, data_dir)
