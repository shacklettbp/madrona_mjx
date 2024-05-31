import math
import sys
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable
import os
from functools import partial

import jax
from jax import random, numpy as jnp

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

class BatchRenderer:
  def __init__(
      self,
      mjx_sys, 
      gpu_id,
      num_worlds,
      batch_render_view_width,
      batch_render_view_height,
      viz_gpu_hdls=None,
  ):
    mesh_verts = mjx_sys.mesh_vert
    mesh_faces = mjx_sys.mesh_face
    mesh_vert_offsets = mjx_sys.mesh_vertadr
    mesh_face_offsets = mjx_sys.mesh_faceadr
    geom_types = mjx_sys.geom_type
    geom_data_ids = mjx_sys.geom_dataid
    geom_sizes = jax.device_get(mjx_sys.geom_size)
    num_cams = mjx_sys.ncam

    # Initialize madrona simulator
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
        visualizer_gpu_handles = viz_gpu_hdls,
    )

    init_prim_fn, render_prim_fn = _setup_jax_primitives(
        self.madrona, num_worlds, geom_sizes.shape[0], num_cams,
        batch_render_view_width, batch_render_view_height)

    @jax.vmap
    def mult_quat(u, v):
      return jnp.array([
          u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
          u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
          u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
          u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
      ])

    def normalize_quat(q):
      n = jnp.linalg.norm(q, ord=2, axis=-1, keepdims=True)
      return q / (n + 1e-6 * (n == 0.0))

    # MJX computes this internally but unfortunately transforms the result to
    # a matrix, Madrona needs a quaternion
    def compute_geom_quats(state, m):
      xquat = state.xquat

      world_quat = state.xquat[m.geom_bodyid]
      local_quat = m.geom_quat

      return normalize_quat(mult_quat(world_quat, local_quat))

    def compute_cam_quats(state, m):
      xquat = state.xquat

      world_quat = state.xquat[m.cam_bodyid]
      local_quat = m.cam_quat

      return normalize_quat(mult_quat(world_quat, local_quat))

    @jax.vmap
    def compute_transforms(state):
      return compute_geom_quats(state, m), compute_cam_quats(state, m)

    @jax.jit
    def render_init(pipeline_state):
      geom_quat, cam_quat = jax.jit(compute_transforms)(
          pipeline_state)
       
      render_token = jnp.array((), jnp.bool)

      init_rgb, init_depth, render_token = init_prim_fn(
          render_token,
          pipeline_state.geom_xpos,
          geom_quat,
          pipeline_state.cam_xpos,
          cam_quat)

      return render_token, init_rgb, init_depth

    @jax.jit
    def render_fn(render_token, mjx_state):
      geom_quat, cam_quat = compute_transforms(pipeline_state)

      return render_prim_fn(render_token,
                            pipeline_state.geom_xpos,
                            geom_quat,
                            pipeline_state.cam_xpos,
                            cam_quat)

    self.init_fn = render_init
    self.render_fn = render_fn

  def init(self, pipeline_state):
    rgb, depth, render_token = self.init_fn(pipeline_state)

    return render_token, rgb, depth

  def render(self, render_token, mjx_state):
    rgb, depth, render_token = self.render_fn(render_token, pipeline_state)

    return render_token, rgb, depth



def _setup_jax_primitives(renderer, num_worlds, num_geoms, num_cams,
                          render_width, render_height):
  custom_call_platform = 'gpu'
  renderer_encode, init_custom_call_capsule, render_custom_call_capsule = renderer.xla_entries()

  renderer_inputs = [
      jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 3),
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 4),
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 3),
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 4),
                           dtype=jnp.float32),
  ]

  renderer_outputs = [
      jax.ShapeDtypeStruct(
          shape=(num_worlds, num_cams, render_height, render_width, 4),
          dtype=jnp.uint8,
      ),
      jax.ShapeDtypeStruct(
          shape=(num_worlds, num_cams, render_height, render_width, 1),
          dtype=jnp.float32,
      ),
      jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
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

