"""Variation of mjx testspeed for benchmarking batch renderings"""

import os
import time
from typing import Tuple
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src.test_util import _measure
from mujoco.mjx._src import io
from mujoco.mjx._src import forward
from etils import epath
import functools

from madrona_mjx.renderer import BatchRenderer

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mjcf', type=str, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--use-raytracer', action='store_true')

arg_parser.add_argument('--nstep', type=int, default=1000)
arg_parser.add_argument('--unroll', type=int, default=1)
arg_parser.add_argument('--solver', type=str, default='newton', choices=['cg', 'newton'])
arg_parser.add_argument('--iterations', type=int, default=1)
arg_parser.add_argument('--ls_iterations', type=int, default=4)

args = arg_parser.parse_args()


def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.2)


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

def benchmark(
    m: mujoco.MjModel,
    nstep: int = 1000,
    batch_size: int = 1024,
    unroll_steps: int = 1,
    solver: str = 'newton',
    iterations: int = 1,
    ls_iterations: int = 4,
    gpu_id: int = 0,
    batch_render_view_width: int = 64,
    batch_render_view_height: int = 64,
    use_raytracer: bool = False,
) -> Tuple[float, float, int]:
  """Benchmark a model."""

  xla_flags = os.environ.get('XLA_FLAGS', '')
  xla_flags += ' --xla_gpu_triton_gemm_any=True'
  os.environ['XLA_FLAGS'] = xla_flags

  m.opt.solver = {
      'cg': mujoco.mjtSolver.mjSOL_CG,
      'newton': mujoco.mjtSolver.mjSOL_NEWTON,
  }[solver.lower()]
  m.opt.iterations = iterations
  m.opt.ls_iterations = ls_iterations
  m = io.put_model(m)

  renderer = BatchRenderer(
    m, gpu_id, batch_size, 
    batch_render_view_width, batch_render_view_width,
    np.array([0, 1, 2]), False, args.use_raytracer,
    None)

  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)

  def dr(sys, rng):
    """Randomizes the mjx.Model."""
    @jax.vmap
    def rand(rng):
      rng, color_rng = jax.random.split(rng, 2)
      new_color = jax.random.uniform(color_rng, (1,), minval=0.0, maxval=0.4)
      geom_rgba = sys.geom_rgba.at[0, 0:1].set(new_color)
      geom_matid = sys.geom_matid.at[:].set(-1)
      geom_matid = geom_matid.at[0].set(-2)

      return geom_rgba, geom_matid

    geom_rgba, geom_matid = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_rgba': 0,
        'geom_matid': 0,
    })

    sys = sys.tree_replace({
        'geom_rgba': geom_rgba,
        'geom_matid': geom_matid,
    })

    return sys, in_axes

  randomization_rng = jax.random.split(rng, args.num_worlds)
  v_randomization_fn = functools.partial(dr, rng=randomization_rng)
  
  v_mjx_model, v_in_axes = v_randomization_fn(m)

  def init(rng, sys):
    def init_(rng, sys):
      data = mjx.make_data(sys)
      data.replace(qpos=0.01 * jax.random.uniform(rng, shape=(sys.nq,)))
      data = mjx.forward(sys, data)
      render_token, rgb, depth = renderer.init(data, sys)
      return data, render_token, rgb, depth
    return jax.vmap(init_, in_axes=[0, v_in_axes])(rng, sys)

  v_mjx_data, render_token, rgb, depth = init(jp.asarray(key), v_mjx_model)
  jax.block_until_ready(v_mjx_data)

  @jax.jit
  def unroll(v_data):
    @jax.vmap
    def step(d, _):
      d = forward.step(m, d)
      _, rgb, depth = renderer.render(render_token, d)
      return d, None
    d, _ = jax.lax.scan(step, v_data, None, length=nstep, unroll=unroll_steps)

    return d

  jit_time, run_time = _measure(unroll, v_mjx_data)
  steps = nstep * batch_size

  return jit_time, run_time, steps


if __name__ == '__main__':
  """Benchmark a model."""
  model = load_model(args.mjcf)
  
  print(f'Rolling out {args.nstep} steps at dt = {model.opt.timestep:.3f}...')
  jit_time, run_time, steps = benchmark(
      m=model,
      nstep=args.nstep,
      batch_size=args.num_worlds,
      unroll_steps=args.unroll,
      solver=args.solver,
      iterations=args.iterations,
      ls_iterations=args.ls_iterations,
      gpu_id=args.gpu_id,
      batch_render_view_width=args.batch_render_view_width,
      batch_render_view_height=args.batch_render_view_height,
      use_raytracer=args.use_raytracer
  )

  print(f"""
  Summary for {args.num_worlds} parallel rollouts

    Total JIT time: {jit_time:.2f} s
    Total simulation time: {run_time:.2f} s
    Total steps per second: { steps / run_time:.0f}
    Total realtime factor: { steps * model.opt.timestep / run_time:.2f} x
    Total time per step: { 1e6 * run_time / steps:.2f} µs""")