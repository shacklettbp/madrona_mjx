import math
import sys
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union
import os
from functools import partial

import jax
from jax import random, numpy as jnp
import numpy as np
from jax import core, dtypes
from jax.core import ShapedArray, Effect
from jax.lib import xla_client
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, dtype_to_ir_type
from jaxlib.hlo_helpers import custom_call
from jax._src import effects
from jax._src.lib.mlir.dialects import hlo

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
    def render_init(mjx_state):
      geom_quat, cam_quat = jax.jit(compute_transforms)(
          mjx_state.pipeline_state)

      init_prim_fn(mjx_state.pipeline_state.geom_xpos,
          geom_quat,
          mjx_state.pipeline_state.cam_xpos,
          cam_quat)

    render_init(init_mjx_state)

    @jax.jit
    def render_fn(mjx_state):
      geom_quat, cam_quat = compute_transforms(mjx_state.pipeline_state)

      return render_prim_fn(mjx_state.pipeline_state.geom_xpos,
                            geom_quat,
                            mjx_state.pipeline_state.cam_xpos,
                            cam_quat)

    self.render_fn = render_fn

  def render(self, mjx_state):
    return self.render_fn(mjx_state)


def _setup_jax_primitives(renderer, num_worlds, num_geoms, num_cams,
                          render_width, render_height):
  custom_call_platform = 'gpu'
  renderer_encode, init_custom_call_capsule, render_custom_call_capsule = renderer.xla_entries()

  renderer_inputs = [
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
  ]

  custom_call_prefix = f"{type(renderer).__name__}_{id(renderer)}"
  init_custom_call_name = f"{custom_call_prefix}_init"
  render_custom_call_name = f"{custom_call_prefix}_render"

  class RenderEffect(Effect):
    __str__ = lambda self: custom_call_prefix

  mlir.lowerable_effects.add_type(RenderEffect)
  effects.ordered_effects.add_type(RenderEffect)
  effects.control_flow_allowed_effects.add_type(RenderEffect)
  _RenderEffect = RenderEffect()

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
  
  # Below code uses ordered effects, which is internal logic taken from
  # jax io_callback code and emit_python_callback code. The idea is a
  # token is threaded through the custom_call, which preserves ordering and
  # prevents sim_render calls from being elided if their outputs aren't used.
  # This code deviates slightly from the jax convention which is to put
  # the token in the first input / output on GPU. Instead, we put the token
  # in the first input and the *last* output, which means we can just skip
  # the first buffer passed to the custom call target (the input token)
  # and write to the rest of the buffers normally, leaving the final token
  # output buffer untouched.
  
  def _prepend_token_to_inputs(types, layouts):
    return [hlo.TokenType.get(), *types], [(), *layouts]
  
  def _append_token_to_results(types, layouts):
    return [*types, hlo.TokenType.get()], [*layouts, ()]
  
  def _init_lowering(ctx, *flattened_inputs):
    # It seems like the input token shouldn't be necessary for init,
    # but using it allows us to keep parity between init and render
    # for gpuEntryFn, which skips the first buffer input.
    token = ctx.tokens_in.get(_RenderEffect)[0]
    inputs = [token, *flattened_inputs]

    input_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
    input_layouts = [_row_major_layout(t.shape) for t in input_types]
    input_types, input_layouts = _prepend_token_to_inputs(
        input_types, input_layouts)
  
    result_types = _lower_shape_dtypes(renderer_outputs)
    result_layouts = [_row_major_layout(t.shape) for t in result_types]
  
    result_types, result_layouts = _append_token_to_results(
        result_types, result_layouts)
  
    results = custom_call(
        init_custom_call_name,
        backend_config=renderer_encode,
        operands=inputs,
        operand_layouts=input_layouts,
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results
  
    *results, token = results
    ctx.set_tokens_out(mlir.TokenSet({_RenderEffect: (token,)}))
    return results
  
  def _init_abstract(*inputs):
    return _shape_dtype_to_abstract_vals(renderer_outputs), {_RenderEffect}
  
  def _render_lowering(ctx, *flattened_inputs):
    token = ctx.tokens_in.get(_RenderEffect)[0]
    inputs = [token, *flattened_inputs]
  
    input_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
    input_layouts = [_row_major_layout(t.shape) for t in input_types]
    input_types, input_layouts = _prepend_token_to_inputs(
        input_types, input_layouts)
  
    result_types = _lower_shape_dtypes(renderer_outputs)
    result_layouts = [_row_major_layout(t.shape) for t in result_types]
    result_types, result_layouts = _append_token_to_results(
        result_types, result_layouts)
  
    results = custom_call(
        render_custom_call_name,
        backend_config=renderer_encode,
        operands=inputs,
        operand_layouts=input_layouts,
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results
  
    *results, token = results
    ctx.set_tokens_out(mlir.TokenSet({_RenderEffect: (token,)}))
  
    return results
  
  def _render_abstract(*inputs):
    return _shape_dtype_to_abstract_vals(renderer_outputs), {_RenderEffect}
  
  
  _init_primitive = core.Primitive(init_custom_call_name)
  _init_primitive.multiple_results = True
  _init_primitive.def_impl(partial(xla.apply_primitive, _init_primitive))
  _init_primitive.def_effectful_abstract_eval(_init_abstract)
  
  mlir.register_lowering(
      _init_primitive,
      _init_lowering,
      platform=custom_call_platform,
  )
  
  _render_primitive = core.Primitive(render_custom_call_name)
  _render_primitive.multiple_results = True
  _render_primitive.def_impl(partial(xla.apply_primitive, _render_primitive))
  _render_primitive.def_effectful_abstract_eval(_render_abstract)
  
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
