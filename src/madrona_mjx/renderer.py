import sys
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable
import os
from functools import partial

import jax
from jax import random, numpy as jp

import numpy as np
from jax import core, dtypes
from jax.core import ShapedArray
from jax.lib import xla_client
from jax.interpreters import batching
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, dtype_to_ir_type
from jaxlib.hlo_helpers import custom_call

from madrona_mjx._madrona_mjx_batch_renderer import MadronaBatchRenderer
from madrona_mjx._madrona_mjx_batch_renderer.madrona import ExecMode

from mujoco.mjx._src import math
from mujoco.mjx._src import io
from mujoco.mjx._src import support

def mat_to_quat(mat):
  """Converts 3D rotation matrix to quaternion."""
  # this is a hack to avoid reimplementing logic in mjx.camlight for quats
  mat = mat.flatten()

  q0 = 0.5 * jp.sqrt(1 + mat[0] + mat[4] + mat[8])
  quat0 = jp.array([
      q0,
      0.25 * (mat[7] - mat[5]) / q0,
      0.25 * (mat[2] - mat[6]) / q0,
      0.25 * (mat[3] - mat[1]) / q0
  ])

  q1 = 0.5 * jp.sqrt(1 + mat[0] - mat[4] - mat[8])
  quat1 = jp.array([
      q1,
      0.25 * (mat[7] - mat[5]) / q1,
      0.25 * (mat[1] + mat[3]) / q1,
      0.25 * (mat[2] + mat[6]) / q1,
  ])

  q2 = 0.5 * jp.sqrt(1 - mat[0] + mat[4] - mat[8])
  quat2 = jp.array([
      q2,
      0.25 * (mat[2] - mat[6]) / q2,
      0.25 * (mat[1] + mat[3]) / q2,
      0.25 * (mat[5] + mat[7]) / q2,
  ])

  q3 = 0.5 * jp.sqrt(1 - mat[0] - mat[4] + mat[8])
  quat3 = jp.array([
      q3,
      0.25 * (mat[3] - mat[1]) / q3,
      0.25 * (mat[2] + mat[6]) / q3,
      0.25 * (mat[5] + mat[7]) / q3,
  ])

  quat0_cond = (mat[0] + mat[4] + mat[8]) > 0
  quat1_cond = (mat[0] > mat[4]) & (mat[0] > mat[8])
  quat2_cond = mat[4] > mat[8]

  quat = jp.where(quat0_cond, quat0, quat3)
  quat = jp.where(~quat0_cond & quat1_cond, quat1, quat)
  quat = jp.where(~quat0_cond & ~quat1_cond & quat2_cond, quat2, quat)

  return quat

class BatchRenderer:
  """Wraps MJX Model around MadronaBatchRenderer."""

  def __init__(
      self,
      m,
      gpu_id,
      num_worlds,
      batch_render_view_width,
      batch_render_view_height,
      add_cam_debug_geo=False,
      viz_gpu_hdls=None,
  ):
    mesh_verts = m.mesh_vert
    mesh_faces = m.mesh_face
    mesh_vert_offsets = m.mesh_vertadr
    mesh_face_offsets = m.mesh_faceadr
    geom_types = m.geom_type
    # TODO: filter geom groups
    geom_data_ids = m.geom_dataid
    geom_sizes = jax.device_get(m.geom_size)
    # TODO: filter for camera ids
    num_cams = m.ncam

    self.madrona = MadronaBatchRenderer(
        exec_mode = ExecMode.CUDA,
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
        add_cam_debug_geo=add_cam_debug_geo,
        visualizer_gpu_handles = viz_gpu_hdls,
    )

    init_fn, render_fn = _setup_jax_primitives(
        self.madrona, num_worlds, geom_sizes.shape[0], num_cams,
        batch_render_view_width, batch_render_view_height)

    self.m = m
    self.init_prim_fn = init_fn
    self.render_prim_fn = render_fn

  def get_geom_quat(self, state):
    to_global = jax.vmap(math.quat_mul)
    geom_quat = to_global(
      state.xquat[self.m.geom_bodyid],
      self.m.geom_quat
    )
    return geom_quat

  def get_cam_quat(self, state):
      def to_quat(mat):
        q = mat_to_quat(mat)

        to_y_fwd = jp.array([0.7071068, -0.7071068, 0, 0], jp.float32)

        q = math.quat_mul(q, to_y_fwd)

        return math.normalize(q)

      return jax.vmap(to_quat)(state.cam_xmat)

  def init(self, state):
    geom_quat = self.get_geom_quat(state)
    cam_quat = self.get_cam_quat(state)

    render_token = jp.array((), jp.bool)

    init_rgb, init_depth, render_token = self.init_prim_fn(
        render_token,
        state.geom_xpos,
        geom_quat,
        state.cam_xpos,
        cam_quat)

    return render_token, init_rgb, init_depth

  def render(self, render_token, state):
    geom_pos = state.geom_xpos
    cam_pos = state.cam_xpos
    geom_quat = self.get_geom_quat(state)
    cam_quat = self.get_cam_quat(state)

    render_token = jp.array((), jp.bool)

    rgb, depth, render_token = self.render_prim_fn(render_token,
        geom_pos, geom_quat, cam_pos, cam_quat)

    return render_token, rgb, depth



def _setup_jax_primitives(renderer, num_worlds, num_geoms, num_cams,
                          render_width, render_height):
  custom_call_platform = 'gpu'
  renderer_encode, init_custom_call_capsule, render_custom_call_capsule = renderer.xla_entries()

  renderer_inputs = [
      jax.ShapeDtypeStruct(shape=(), dtype=jp.bool),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 3),
                           dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 4),
                           dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 3),
                           dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 4),
                           dtype=jp.float32),
  ]

  renderer_outputs = [
      jax.ShapeDtypeStruct(
          shape=(num_worlds, num_cams, render_height, render_width, 4),
          dtype=jp.uint8,
      ),
      jax.ShapeDtypeStruct(
          shape=(num_worlds, num_cams, render_height, render_width, 1),
          dtype=jp.float32,
      ),
      jax.ShapeDtypeStruct(shape=(), dtype=jp.bool),
  ]

  custom_call_prefix = f"{type(renderer).__name__}_{id(renderer)}"
  init_custom_call_name = f"{custom_call_prefix}_init"
  render_custom_call_name = f"{custom_call_prefix}_render"

  xla_client.register_custom_call_target(
    init_custom_call_name, init_custom_call_capsule,
    platform=custom_call_platform)

  xla_client.register_custom_call_target(
    render_custom_call_name, render_custom_call_capsule,
    platform=custom_call_platform)

  def _row_major_layout(shape):
    return tuple(range(len(shape) -1, -1, -1))

  def _shape_dtype_to_abstract_vals(vs):
    return tuple(ShapedArray(v.shape, v.dtype) for v in vs)

  def _lower_shape_dtypes(shape_dtypes):
    return [ir.RankedTensorType.get(i.shape, dtype_to_ir_type(i.dtype))
        for i in shape_dtypes]

  def _init_lowering(ctx, *inputs):
    input_types = [ir.RankedTensorType(i.type) for i in inputs]
    input_layouts = [_row_major_layout(t.shape) for t in input_types]

    result_types = _lower_shape_dtypes(renderer_outputs)
    result_layouts = [_row_major_layout(t.shape) for t in result_types]

    results = custom_call(
        init_custom_call_name,
        backend_config=renderer_encode,
        operands=inputs,
        operand_layouts=input_layouts,
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results

    return results

  def _init_abstract(*inputs):
    return _shape_dtype_to_abstract_vals(renderer_outputs)

  def _render_lowering(ctx, *inputs):
    input_types = [ir.RankedTensorType(i.type) for i in inputs]
    input_layouts = [_row_major_layout(t.shape) for t in input_types]

    result_types = _lower_shape_dtypes(renderer_outputs)
    result_layouts = [_row_major_layout(t.shape) for t in result_types]

    results = custom_call(
        render_custom_call_name,
        backend_config=renderer_encode,
        operands=inputs,
        operand_layouts=input_layouts,
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results

    return results

  def _render_abstract(*inputs):
    return _shape_dtype_to_abstract_vals(renderer_outputs)

  _init_primitive = core.Primitive(init_custom_call_name)
  _init_primitive.multiple_results = True
  _init_primitive_impl = partial(xla.apply_primitive, _init_primitive)
  _init_primitive.def_impl(_init_primitive_impl)
  _init_primitive.def_abstract_eval(_init_abstract)

  mlir.register_lowering(
      _init_primitive,
      _init_lowering,
      platform=custom_call_platform,
  )

  _render_primitive = core.Primitive(render_custom_call_name)
  _render_primitive.multiple_results = True
  _render_primitive_impl = partial(xla.apply_primitive, _render_primitive)
  _render_primitive.def_impl(_render_primitive_impl)
  _render_primitive.def_abstract_eval(_render_abstract)

  def _init_prim_batch(vector_arg_values, batch_axes):
    assert all(b == batch_axes[1] for b in batch_axes[1:])
    result_axes = [batch_axes[1], batch_axes[1], batch_axes[0]]
    return _init_primitive_impl(*vector_arg_values), result_axes

  def _render_prim_batch(vector_arg_values, batch_axes):
    assert all(b == batch_axes[1] for b in batch_axes[1:])
    result_axes = [batch_axes[1], batch_axes[1], batch_axes[0]]
    return _render_primitive_impl(*vector_arg_values), result_axes

  batching.primitive_batchers[_init_primitive] = _init_prim_batch
  batching.primitive_batchers[_render_primitive] = _render_prim_batch

  mlir.register_lowering(
      _render_primitive,
      _render_lowering,
      platform=custom_call_platform,
  )

  @jax.jit
  def init_fn(*args):
      return _init_primitive.bind(*args)

  @jax.jit
  def render_fn(*args):
      return _render_primitive.bind(*args)

  return init_fn, render_fn

