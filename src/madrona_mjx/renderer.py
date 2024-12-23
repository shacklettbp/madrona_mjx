from functools import partial
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
from jax import core
from jax import dtypes
from jax import lax
from jax import numpy as jp
from jax import random
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import dtype_to_ir_type
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from mujoco.mjx._src import io
from mujoco.mjx._src import math
from mujoco.mjx._src import support
import numpy as np

from madrona_mjx._madrona_mjx_batch_renderer import MadronaBatchRenderer
from madrona_mjx._madrona_mjx_batch_renderer.madrona import ExecMode


def mat_to_quat(mat):
  """Converts 3D rotation matrix to quaternion."""
  # this is a hack to avoid reimplementing logic in mjx.camlight for quats
  a = mat[:, 0]
  b = mat[:, 1]
  c = mat[:, 2]

  # Converted from madrona::Quat::fromBasis to jax.
  #
  # Originally based on glm::quat_cast:
  # Copyright (c) 2005 - G-Truc Creation
  # Permission is hereby granted, free of charge, to any person obtaining a copy
  # of this software and associated documentation files (the "Software"), to deal
  # in the Software without restriction, including without limitation the rights
  # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  # copies of the Software, and to permit persons to whom the Software is
  # furnished to do so, subject to the following conditions:
  #
  # The above copyright notice and this permission notice shall be included in
  # all copies or substantial portions of the Software.
  #
  # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  # THE SOFTWARE.

  four_sq_minus1 = jp.array(
      [
          a[0] + b[1] + c[2],  # w
          a[0] - b[1] - c[2],  # x
          b[1] - a[0] - c[2],  # y
          c[2] - a[0] - b[1],  # z,
      ],
      jp.float32,
  )

  biggest_index = jp.argmax(four_sq_minus1)
  biggest_val = jp.sqrt(four_sq_minus1[biggest_index] + 1) * 0.5
  mult = 0.25 / biggest_val

  def big_w(biggest, mult, a, b, c):
    return jp.array(
        [
            biggest,
            (b[2] - c[1]) * mult,
            (c[0] - a[2]) * mult,
            (a[1] - b[0]) * mult,
        ],
        jp.float32,
    )

  def big_x(biggest, mult, a, b, c):
    return jp.array(
        [
            (b[2] - c[1]) * mult,
            biggest,
            (a[1] + b[0]) * mult,
            (c[0] + a[2]) * mult,
        ],
        jp.float32,
    )

  def big_y(biggest, mult, a, b, c):
    return jp.array(
        [
            (c[0] - a[2]) * mult,
            (a[1] + b[0]) * mult,
            biggest,
            (b[2] + c[1]) * mult,
        ],
        jp.float32,
    )

  def big_z(biggest, mult, a, b, c):
    return jp.array(
        [
            (a[1] - b[0]) * mult,
            (c[0] + a[2]) * mult,
            (b[2] + c[1]) * mult,
            biggest,
        ],
        jp.float32,
    )

  quat = lax.switch(
      biggest_index, [big_w, big_x, big_y, big_z], biggest_val, mult, a, b, c
  )

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
      enabled_geom_groups=np.array([0, 1, 2]),
      add_cam_debug_geo=False,
      use_rasterizer=False,
      viz_gpu_hdls=None,
  ):
    mesh_verts = m.mesh_vert
    mesh_faces = m.mesh_face
    mesh_vert_offsets = m.mesh_vertadr
    mesh_face_offsets = m.mesh_faceadr
    mesh_texcoords = m.mesh_texcoord
    mesh_texcoord_offsets = m.mesh_texcoordadr
    mesh_texcoord_num = m.mesh_texcoordnum
    geom_types = m.geom_type
    geom_groups = m.geom_group
    geom_data_ids = m.geom_dataid
    geom_sizes = jax.device_get(m.geom_size)
    geom_mat_ids = jax.device_get(m.geom_matid)
    geom_rgba = jax.device_get(m.geom_rgba)
    mat_rgba = jax.device_get(m.mat_rgba)
    num_lights = m.nlight
    num_cams = m.ncam
    assert num_cams > 0  # Must have at least one camera for Madrona to work!

    mat_tex_ids = m.mat_texid
    tex_data = m.tex_data
    # add 255 every third element to create 4 channel rgba texture
    tex_data = np.insert(
        tex_data, np.arange(3, tex_data.shape[0], 3), 255, axis=0
    )
    tex_offsets = m.tex_adr
    tex_widths = m.tex_width
    tex_heights = m.tex_height
    tex_nchans = m.tex_nchannel

    self.madrona = MadronaBatchRenderer(
        gpu_id=gpu_id,
        mesh_vertices=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_vertex_offsets=mesh_vert_offsets,
        mesh_face_offsets=mesh_face_offsets,
        mesh_texcoords=mesh_texcoords,
        mesh_texcoord_offsets=mesh_texcoord_offsets,
        mesh_texcoord_num=mesh_texcoord_num,
        geom_types=geom_types,
        geom_groups=geom_groups,
        geom_data_ids=geom_data_ids,
        geom_sizes=geom_sizes,
        geom_mat_ids=geom_mat_ids,
        geom_rgba=geom_rgba,
        mat_rgba=mat_rgba,
        mat_tex_ids=mat_tex_ids,
        tex_data=tex_data,
        tex_offsets=tex_offsets,
        tex_widths=tex_widths,
        tex_heights=tex_heights,
        tex_nchans=tex_nchans,
        num_lights=num_lights,
        num_cams=num_cams,
        num_worlds=num_worlds,
        batch_render_view_width=batch_render_view_width,
        batch_render_view_height=batch_render_view_height,
        enabled_geom_groups=enabled_geom_groups,
        add_cam_debug_geo=add_cam_debug_geo,
        use_rt=not use_rasterizer,
        visualizer_gpu_handles=viz_gpu_hdls,
    )

    init_fn, render_fn = _setup_jax_primitives(
        self.madrona,
        num_worlds,
        geom_sizes.shape[0],
        num_cams,
        batch_render_view_width,
        batch_render_view_height,
    )

    self.m = m
    self.init_prim_fn = init_fn
    self.render_prim_fn = render_fn

  def get_geom_quat(self, state):
    to_global = jax.vmap(math.quat_mul)
    geom_quat = to_global(state.xquat[self.m.geom_bodyid], self.m.geom_quat)
    return geom_quat

  def get_cam_quat(self, state):
    def to_quat(mat):
      q = mat_to_quat(mat)

      to_y_fwd = jp.array([0.7071068, -0.7071068, 0, 0], jp.float32)

      q = math.quat_mul(q, to_y_fwd)

      return math.normalize(q)

    return jax.vmap(to_quat)(state.cam_xmat)

  def adjust_scale(self, geom_size, geom_type):
    """Returns the adjusted madrona scale of the geometry based on geom_type."""

    def adjust(size, gtype):
      x, y, z = size
      # Plane
      size = size.at[:].set(
          jp.where(gtype == 0, jp.array([x * 2, y * 2, 1], jp.float32), size)
      )
      # Heightfields - Currently not supported
      size = size.at[:].set(
          jp.where(gtype == 1, jp.array([1, 1, 1], jp.float32), size)
      )
      # Sphere
      size = size.at[:].set(
          jp.where(gtype == 2, jp.array([x, x, x], jp.float32), size)
      )
      # Capsule - Resize not supported
      size = size.at[:].set(
          jp.where(gtype == 3, jp.array([1, 1, 1], jp.float32), size)
      )
      # Ellipsoid - Currently not supported
      size = size.at[:].set(
          jp.where(gtype == 4, jp.array([1, 1, 1], jp.float32), size)
      )
      # Cylinder
      size = size.at[:].set(
          jp.where(gtype == 5, jp.array([x, y, 1], jp.float32), size)
      )
      # Box
      size = size.at[:].set(
          jp.where(gtype == 6, jp.array([x, y, z], jp.float32), size)
      )
      # Mesh - Resize not supported
      size = size.at[:].set(
          jp.where(gtype == 7, jp.array([1, 1, 1], jp.float32), size)
      )
      return size

    return jax.vmap(adjust)(geom_size, geom_type)

  def init(self, state, model):
    geom_quat = self.get_geom_quat(state)
    cam_quat = self.get_cam_quat(state)
    geom_rgba_uint = jp.array(model.geom_rgba * 255, jp.uint32)
    geom_size = self.adjust_scale(model.geom_size, model.geom_type)

    def rgb2int(rgb):
      color = 0
      color = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
      return color

    rgb_uint32 = jax.vmap(rgb2int)(geom_rgba_uint)

    render_token = jp.array((), jp.bool)

    init_rgb, init_depth, render_token = self.init_prim_fn(
        render_token,
        state.geom_xpos,
        geom_quat,
        state.cam_xpos,
        cam_quat,
        model.geom_matid,
        rgb_uint32,
        geom_size,
        model.light_pos,
        model.light_dir,
        model.light_directional,
        model.light_castshadow,
        model.light_cutoff,
    )

    return render_token, init_rgb, init_depth

  def render(self, render_token, state):
    geom_pos = state.geom_xpos
    cam_pos = state.cam_xpos
    geom_quat = self.get_geom_quat(state)
    cam_quat = self.get_cam_quat(state)

    rgb, depth, render_token = self.render_prim_fn(
        render_token, geom_pos, geom_quat, cam_pos, cam_quat
    )

    return render_token, rgb, depth


def _setup_jax_primitives(
    renderer, num_worlds, num_geoms, num_cams, render_width, render_height
):
  custom_call_platform = "gpu"
  renderer_encode, init_custom_call_capsule, render_custom_call_capsule = (
      renderer.xla_entries()
  )

  renderer_inputs = [
      jax.ShapeDtypeStruct(shape=(), dtype=jp.bool),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 3), dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_geoms, 4), dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 3), dtype=jp.float32),
      jax.ShapeDtypeStruct(shape=(num_worlds, num_cams, 4), dtype=jp.float32),
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
      init_custom_call_name,
      init_custom_call_capsule,
      platform=custom_call_platform,
  )

  xla_client.register_custom_call_target(
      render_custom_call_name,
      render_custom_call_capsule,
      platform=custom_call_platform,
  )

  def _row_major_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))

  def _shape_dtype_to_abstract_vals(vs):
    return tuple(ShapedArray(v.shape, v.dtype) for v in vs)

  def _lower_shape_dtypes(shape_dtypes):
    return [
        ir.RankedTensorType.get(i.shape, dtype_to_ir_type(i.dtype))
        for i in shape_dtypes
    ]

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
    # assert all(b == batch_axes[1] for b in batch_axes[1:])
    batch_axes = list(batch_axes)
    for i in range(1, len(batch_axes)):
      if batch_axes[i] != 0:
        print("Inferred batch not found, overriding manually")
        batch_axes[i] = 0
    batch_dims = vector_arg_values[1].shape[:-2]
    # TODO: Replace hacks on these batch dimension checks and reshapes
    if len(batch_dims) > 1:
      #num_worlds = np.prod(batch_dims)
      #params = tuple(
      #    jp.reshape(v, (num_worlds,) + v.shape[len(batch_dims):])
      #    for v in vector_arg_values[1:])
      #vector_arg_values = vector_arg_values[:1] + params
      num_worlds = np.prod(batch_dims)
      params = tuple(
          jp.reshape(v, (num_worlds,) + v.shape[len(batch_dims) :])
          for v in vector_arg_values[1:5]
      )
      vector_arg_values = vector_arg_values[:1] + params + vector_arg_values[5:]
    result = _init_primitive_impl(*vector_arg_values)
    result_axes = [batch_axes[1], batch_axes[1], batch_axes[0]]
    if len(batch_dims) > 1:
      params = [jp.reshape(v, batch_dims + v.shape[1:]) for v in result[:-1]]
      result = params + result[-1:]
    return result, result_axes

  def _render_prim_batch(vector_arg_values, batch_axes):
    # assert all(b == batch_axes[1] for b in batch_axes[1:])
    batch_axes = list(batch_axes)
    for i in range(1, len(batch_axes)):
      if batch_axes[i] != batch_axes[1]:
        print("Inferred batch not found, overriding manually")
        batch_axes[i] = 0
    batch_dims = vector_arg_values[1].shape[:-2]
    if len(batch_dims) > 1:
      num_worlds = np.prod(batch_dims)
      params = tuple(
          jp.reshape(v, (num_worlds,) + v.shape[len(batch_dims) :])
          for v in vector_arg_values[1:]
      )
      vector_arg_values = vector_arg_values[:1] + params
    result = _render_primitive_impl(*vector_arg_values)
    result_axes = [batch_axes[1], batch_axes[1], batch_axes[0]]
    if len(batch_dims) > 1:
      params = [jp.reshape(v, batch_dims + v.shape[1:]) for v in result[:-1]]
      result = params + result[-1:]
    return result, result_axes

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
