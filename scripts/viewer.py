import os
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from etils import epath

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

  def init(rng):
    def init_(rng):
      mjx_data = mjx.make_data(mjx_model)
      mjx_data.replace(qpos=0.01 * jax.random.uniform(rng, shape=(mjx_model.nq,)))
      mjx_data = mjx.forward(mjx_model, mjx_data)
      render_token, rgb, depth = renderer.init(mjx_data)
      return mjx_data, render_token
    
    return jax.vmap(init_)(rng)

  def step(data, action):
    def step_(data, action):
      data.replace(ctrl=action)
      data = mjx.step(mjx_model, data)
      _, rgb, depth = renderer.render(render_token, data)
      return data, rgb
    return jax.vmap(step_)(data, action)

  init_fn = jax.jit(init)
  step_fn = jax.jit(step)
  mjx_data, render_token = init_fn(jp.asarray(key))

  def vis_step_fn(carry):
    rng, data = carry
    rng, act_rng = jax.random.split(rng)
    ctrl = jax.random.uniform(act_rng, shape=(args.num_worlds, mjx_model.nu))
    data, rgb = step_fn(data, ctrl)
    return rng, data

  visualizer = Visualizer(viz_gpu_state, renderer.madrona)
  visualizer.loop(renderer.madrona, vis_step_fn, (rng, mjx_data))
