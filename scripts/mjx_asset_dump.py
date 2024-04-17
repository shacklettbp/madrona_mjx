import jax
from jax import random, numpy as jp
import numpy as np
import pygltflib

from madrona_mjx import BatchRenderer
import argparse
from time import time

from mjx_env import MJXEnvAndPolicy

import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dump-path', type=str, required=True)

args = arg_parser.parse_args()

mjx_wrapper = MJXEnvAndPolicy.create(random.key(0), 1)
m = mjx_wrapper.env.sys

mesh_verts = m.mesh_vert
mesh_faces = m.mesh_face
mesh_vert_offsets = m.mesh_vertadr
mesh_face_offsets = m.mesh_faceadr
geom_types = m.geom_type
geom_data_ids = m.geom_dataid
geom_sizes = jax.device_get(m.geom_size)

gltf_buffer_views = []
gltf_accessors = []
gltf_meshes = []
gltf_instances = []

for i in range(len(mesh_vert_offsets)):
    vert_offset = mesh_vert_offsets[i]
    face_offset = mesh_face_offsets[i]
    if i == len(mesh_vert_offsets) - 1:
        num_verts = mesh_verts.shape[0] - vert_offset
        num_faces = mesh_faces.shape[0] - face_offset
    else:
        num_verts = mesh_vert_offsets[i + 1] - vert_offset
        num_faces = mesh_face_offsets[i + 1] - face_offset

    vert_offset = int(vert_offset)
    face_offset = int(face_offset)
    num_verts = int(num_verts)
    num_faces = int(num_faces)

    num_bytes_per_face = 12
    num_bytes_per_vert = 12

    gltf_buffer_views.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=num_bytes_per_face * face_offset,
        byteLength=num_bytes_per_face * num_faces,
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))

    gltf_accessors.append(pygltflib.Accessor(
        bufferView=len(gltf_buffer_views) - 1,
        componentType=pygltflib.UNSIGNED_INT,
        count=num_faces * 3,
        type=pygltflib.SCALAR,
        max=[int(mesh_faces.max())],
        min=[int(mesh_faces.min())],
    ))

    gltf_buffer_views.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=num_bytes_per_face * mesh_faces.shape[0] + num_bytes_per_vert * vert_offset,
        byteLength=num_bytes_per_vert * num_verts,
        target=pygltflib.ARRAY_BUFFER,
    ))

    gltf_accessors.append(pygltflib.Accessor(
        bufferView=len(gltf_buffer_views) - 1,
        componentType=pygltflib.FLOAT,
        count=num_verts,
        type=pygltflib.VEC3,
        max=mesh_verts.max(axis=0).tolist(),
        min=mesh_verts.min(axis=0).tolist(),
    ))

    gltf_meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=len(gltf_accessors) - 1),
        indices=len(gltf_accessors) - 2,
        material=0,
    )]))

geom_xpos = jax.device_get(mjx_wrapper.mjx_state.pipeline_state.geom_xpos[0])
geom_xmat = jax.device_get(mjx_wrapper.mjx_state.pipeline_state.geom_xmat[0])

for i in range(len(geom_sizes)):
    geom_types = m.geom_type
    geom_data_ids = m.geom_dataid
    geom_sizes = jax.device_get(m.geom_size)

    geom_type = geom_types[i]
    geom_size = geom_sizes[i]

    pos = geom_xpos[i]
    mat = geom_xmat[i]

    if geom_type == 7:
        gltf_instances.append(pygltflib.Node(
            mesh=int(geom_data_ids[i]),
            matrix=[
                float(mat[0][0]), float(mat[1][0]), float(mat[2][0]), 0,
                float(mat[0][1]), float(mat[1][1]), float(mat[2][1]), 0,
                float(mat[0][2]), float(mat[1][2]), float(mat[2][2]), 0,
                float(pos[0]),    float(pos[1]),    float(pos[2]),    1,
            ],
        ))

# write to gltf
triangles_binary_blob = mesh_faces.flatten().tobytes()
points_binary_blob = mesh_verts.tobytes()

gltf = pygltflib.GLTF2(
    scene=0,
    scenes=[pygltflib.Scene(nodes=list(range(len(gltf_instances))))],
    nodes=gltf_instances,
    meshes=gltf_meshes,
    accessors=gltf_accessors,
    bufferViews=gltf_buffer_views,
    buffers=[
        pygltflib.Buffer(
            byteLength=len(triangles_binary_blob) + len(points_binary_blob)
        )
    ],
    materials=[
        pygltflib.Material(pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )),
    ],
)
gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)

gltf.save(args.dump_path)
