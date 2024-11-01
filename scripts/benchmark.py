"""Variation of mjx testspeed for benchmarking batch renderings"""

import os
from typing import Tuple
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.test_util import _measure
from mujoco.mjx._src import io
from etils import epath

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
limit_jax_mem(0.1)


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

  @jax.pmap
  def init(key):
    key = jax.random.split(key, batch_size // jax.device_count())

    @jax.vmap
    def random_init(key):
      d = io.make_data(m)
      qvel = 0.01 * jax.random.normal(key, shape=(m.nv,))
      d = d.replace(qvel=qvel)
      rt, rgb, depth = renderer.init(mjx_data)
      return d, rt

    return random_init(key)

  key = jax.random.split(jax.random.key(0), jax.device_count())
  d, rt = init(key)
  jax.block_until_ready(d)

  @jax.pmap
  def unroll(d):
    @jax.vmap
    def step(d, _):
      d = forward.step(m, d)
      _, rgb, depth = renderer.render(rt, d)
      return d, rgb, None

    d, rgb, _ = jax.lax.scan(step, d, None, length=nstep, unroll=unroll_steps)

    return d, rgb

  jit_time, run_time = _measure(unroll, d)
  steps = nstep * batch_size

  return jit_time, run_time, steps


if __name__ == '__main__':
  """Benchmark a model."""
  model = load_model(args.mjcf)
  
  print(f'Rolling out {args.nstep} steps at dt = {model.opt.timestep:.3f}...')
  jit_time, run_time, steps = mjx.benchmark(
      model,
      args.nstep,
      args.num_worlds,
      args.unroll,
      args.solver,
      args.iterations,
      args.ls_iterations
  )

  print(f"""
  Summary for {args.num_worlds} parallel rollouts

    Total JIT time: {jit_time:.2f} s
    Total simulation time: {run_time:.2f} s
    Total steps per second: { steps / run_time:.0f}
    Total realtime factor: { steps * model.opt.timestep / run_time:.2f} x
    Total time per step: { 1e6 * run_time / steps:.2f} µs""")