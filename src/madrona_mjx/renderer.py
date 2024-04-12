import math
import sys
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union
import os
from functools import partial

import jax
from jax import random, numpy as jp
import numpy as np
from madrona_mjx._madrona_mjx_batch_renderer import MadronaBatchRenderer, madrona

class BatchRenderer:
  def __init__(
      self,
      mjx_env,
      init_mjx_state,
      gpu_id,
      num_worlds,
      batch_render_view_width,
      batch_render_view_height,
      cpu_madrona,
      viz_gpu_hdls=None,
  ):
    # Initialize madrona simulator

    m = mjx_env.sys

    mesh_verts = m.mesh_vert
    mesh_faces = m.mesh_face
    mesh_vert_offsets = m.mesh_vertadr
    mesh_face_offsets = m.mesh_faceadr
    geom_types = m.geom_type
    geom_data_ids = m.geom_dataid
    geom_sizes = jax.device_get(m.geom_size)
    num_cams = m.ncam

    self.madrona = MadronaBatchRenderer(
        exec_mode = madrona.ExecMode.CPU if cpu_madrona else madrona.ExecMode.CUDA,
        gpu_id = gpu_id,
        mesh_vertices = mesh_verts,
        mesh_faces = mesh_faces,
        mesh_vertex_offsets = mesh_vert_offsets,
        mesh_face_offsets = mesh_face_offsets,
        geom_types = geom_types,
        geom_data_ids = geom_data_ids,
        geom_sizes = geom_sizes,
        num_cams = num_cams,
        num_worlds = num_worlds,
        batch_render_view_width = batch_render_view_width,
        batch_render_view_height = batch_render_view_height,
        visualizer_gpu_handles = viz_gpu_hdls,
    )

    @jax.vmap
    def mult_quat(u, v):
      return jp.array([
          u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
          u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
          u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
          u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
      ])

    # MJX computes this internally but unfortunately transforms the result to
    # a matrix, Madrona needs a quaternion
    def compute_geom_quats(state, m):
      xquat = state.xquat

      world_quat = state.xquat[m.geom_bodyid]
      local_quat = m.geom_quat

      composed = mult_quat(world_quat, local_quat)

      n = jp.linalg.norm(composed, ord=2, axis=-1, keepdims=True)
      return composed / (n + 1e-6 * (n == 0.0))

    def compute_cam_quats(state, m):
      xquat = state.xquat

      world_quat = state.xquat[m.cam_bodyid]
      local_quat = m.cam_quat

      composed = mult_quat(world_quat, local_quat)

      n = jp.linalg.norm(composed, ord=2, axis=-1, keepdims=True)
      return composed / (n + 1e-6 * (n == 0.0))

    def compute_transforms(state):
      return compute_geom_quats(state, m), compute_cam_quats(state, m)

    self.compute_transforms = jax.jit(jax.vmap(compute_transforms))

    geom_quat, cam_quat = self.compute_transforms(init_mjx_state.pipeline_state)

    self.madrona.init(jax.dlpack.to_dlpack(init_mjx_state.pipeline_state.geom_xpos),
                      jax.dlpack.to_dlpack(geom_quat),
                      jax.dlpack.to_dlpack(init_mjx_state.pipeline_state.cam_xpos),
                      jax.dlpack.to_dlpack(cam_quat))

  def render(self, mjx_state):
    geom_quat, cam_quat = self.compute_transforms(mjx_state.pipeline_state)

    self.madrona.render(jax.dlpack.to_dlpack(mjx_state.pipeline_state.geom_xpos),
                        jax.dlpack.to_dlpack(geom_quat),
                        jax.dlpack.to_dlpack(mjx_state.pipeline_state.cam_xpos),
                        jax.dlpack.to_dlpack(cam_quat))

    return mjx_state

