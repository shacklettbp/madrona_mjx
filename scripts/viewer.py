import os
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from etils import epath
import functools

from madrona_mjx.renderer import BatchRenderer
from madrona_mjx.viz import VisualizerGPUState, Visualizer

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mjcf', type=str, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--add-cam-debug-geo', action='store_true')
arg_parser.add_argument('--use-raytracer', action='store_true')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(
  args.window_width, args.window_height, args.gpu_id)

def load_model(path: str):
  path = epath.Path(path)
  xml = path.read_text()
  assets = {}
  for f in path.parent.glob('*.xml'):
    assets[f.name] = f.read_bytes()
    for f in (path.parent / 'assets').glob('*'):
      assets[f.name] = f.read_bytes()
  model = mujoco.MjModel.from_xml_string(xml, assets)
  return model

def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.05)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if __name__ == '__main__':
  
  model = load_model(args.mjcf)
  mjx_model = mjx.put_model(model)

  renderer = BatchRenderer(
    mjx_model, args.gpu_id, args.num_worlds, 
    args.batch_render_view_width, args.batch_render_view_width,
    np.array([0, 1, 2]), args.add_cam_debug_geo, args.use_raytracer,
    viz_gpu_state.get_gpu_handles())

  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)

  def dr(sys, rng):
    """Randomizes the mjx.Model."""
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

      return geom_rgba, geom_matid, geom_size

    geom_rgba, geom_matid, geom_size = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_rgba': 0,
        'geom_matid': 0,
        'geom_size': 0,
    })

    sys = sys.tree_replace({
        'geom_rgba': geom_rgba,
        'geom_matid': geom_matid,
        'geom_size': geom_size,
    })

    return sys, in_axes

  randomization_rng = jax.random.split(rng, args.num_worlds)
  v_randomization_fn = functools.partial(dr, rng=randomization_rng)
  
  v_mjx_model, v_in_axes = v_randomization_fn(mjx_model)

  @jax.jit
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
