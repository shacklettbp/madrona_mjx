"""Viewer for MJX using the Madrona BatchRenderer.

This scripts launches a viewer for MJX using the Madrona BatchRenderer. The
script showcases how to use domain randomization on the MJX model to randomize
the visual properties across worlds.

Usage:
    python viewer.py --mjcf <path_to_mjcf> --num-worlds <num_worlds> \
      --batch-render-view-width <width> --batch-render-view-height <height> \
      --window-width 2000 --window-height 1500 [options]
"""

import argparse
import functools
import os

from etils import epath
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from madrona_mjx.renderer import BatchRenderer
from madrona_mjx.viz import Visualizer
from madrona_mjx.viz import VisualizerGPUState
from madrona_mjx.wrapper import load_model

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mjcf', type=str, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, default=16)
arg_parser.add_argument('--window-width', type=int, default=1500)
arg_parser.add_argument('--window-height', type=int, default=800)
arg_parser.add_argument('--batch-render-view-width', type=int, default=128)
arg_parser.add_argument('--batch-render-view-height', type=int, default=128)
arg_parser.add_argument('--add-cam-debug-geo', action='store_true')
arg_parser.add_argument('--use-rasterizer', action='store_true')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(
    args.window_width, args.window_height, args.gpu_id
)


def limit_jax_mem(limit):
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{limit:.2f}'


limit_jax_mem(0.1)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


def domain_randomize(sys, rng):
  """Randomizes the MJX Model along specified axes.

  Madrona expects certain MjModel axes to be batched so that the buffers can
  be copied to the GPU. Therefore, domain randomization functions must
  randomize geom_rgba, geom_matid, and geom_size so that the buffers are filled.

  If material id is -1, it uses the default material
  that was originally designated during madrona initialization. If material id
  is -2, it uses the color override specified in geom_rgba. Material id can also
  be positive which will set it to a specific material from one of the original
  generated materials during Madrona initialization."""

  @jax.vmap
  def rand(rng):
    rng, color_rng = jax.random.split(rng, 2)

    # A matid of -1 means use default material,
    # matid of -2 means use color override
    geom_matid = sys.geom_matid.at[:].set(-1)
    geom_matid = geom_matid.at[0].set(-2)

    new_color = jax.random.uniform(color_rng, (1,), minval=0.0, maxval=0.4)
    geom_rgba = sys.geom_rgba.at[0, 0:1].set(new_color)
    new_size = jax.random.uniform(color_rng, (3,), minval=1, maxval=5)
    geom_size = sys.geom_size.at[0, 0:3].set(new_size)

    # Lighting randomization
    new_light_pos = jax.random.uniform(
        rng,
        (3,),
        minval=jp.asarray([-0.5, -0.5, 2]),
        maxval=jp.asarray([0.5, 0.5, 2]),
    )
    light_pos = sys.light_pos.at[:].set(new_light_pos)
    light_dir = sys.light_dir.at[:].set(jp.asarray([0, 0, -1]))
    light_directional = sys.light_directional.at[:].set(False)
    light_castshadow = sys.light_castshadow.at[:].set(True)
    light_cutoff = sys.light_cutoff.at[:].set(
        jax.random.uniform(rng, (1,), minval=1, maxval=1.5)
    )
    return (
        geom_rgba,
        geom_matid,
        geom_size,
        light_pos,
        light_dir,
        light_directional,
        light_castshadow,
        light_cutoff,
    )

  (
      geom_rgba,
      geom_matid,
      geom_size,
      light_pos,
      light_dir,
      light_directional,
      light_castshadow,
      light_cutoff,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_directional': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })

  sys = sys.tree_replace({
      'geom_rgba': geom_rgba,
      'geom_matid': geom_matid,
      'geom_size': geom_size,
      'light_pos': light_pos,
      'light_dir': light_dir,
      'light_directional': light_directional,
      'light_castshadow': light_castshadow,
      'light_cutoff': light_cutoff,
  })

  return sys, in_axes


if __name__ == '__main__':
  model = load_model(args.mjcf)
  mjx_model = mjx.put_model(model)

  renderer = BatchRenderer(
      mjx_model,
      args.gpu_id,
      args.num_worlds,
      args.batch_render_view_width,
      args.batch_render_view_width,
      np.array([0, 1, 2]),
      None,
      args.add_cam_debug_geo,
      args.use_rasterizer,
      viz_gpu_state.get_gpu_handles(),
  )

  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)

  randomization_rng = jax.random.split(rng, args.num_worlds)
  v_randomization_fn = functools.partial(
      domain_randomize, rng=randomization_rng
  )
  v_mjx_model, v_in_axes = v_randomization_fn(mjx_model)

  def init(rng, sys):
    def init_(rng, sys):
      data = mjx.make_data(sys)
      data.replace(qpos=0.01 * jax.random.uniform(rng, shape=(sys.nq,)))
      data = mjx.forward(sys, data)
      render_token, rgb, depth = renderer.init(data, sys)
      return data, render_token, rgb, depth

    return jax.vmap(init_, in_axes=[0, v_in_axes])(rng, sys)

  v_mjx_data, render_token, rgb, depth = init(jp.asarray(key), v_mjx_model)

  def step(data, action):
    def step_(data, action):
      data.replace(ctrl=action)
      data = mjx.step(mjx_model, data)
      _, rgb, depth = renderer.render(render_token, data)
      return data, rgb, depth

    return jax.vmap(step_)(data, action)

  step_fn = jax.jit(step)

  def vis_step_fn(carry):
    rng, data = carry
    rng, act_rng = jax.random.split(rng)
    ctrl = jax.random.uniform(act_rng, shape=(args.num_worlds, mjx_model.nu))
    data, rgb, depth = step_fn(data, ctrl)
    return rng, data

  visualizer = Visualizer(viz_gpu_state, renderer.madrona)
  visualizer.loop(renderer.madrona, vis_step_fn, (rng, v_mjx_data))
