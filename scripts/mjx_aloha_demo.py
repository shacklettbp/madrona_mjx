r"""Aloha demo with madrona batch rendering.

This demo requires a patch to the Aloha files:

```sh
pip install brax
pip intall mediapy

cd mujoco_menagerie

patch -p0 < mjx_aloha.patch
patch -p0 < mjx_filtered_cartesian_actuators.patch
patch -p0 < mjx_scene.patch

mv aloha.xml mjx_aloha.xml
mv filtered_cartesian_actuators.xml mjx_filtered_cartesian_actuators.xml
mv scene.xml mjx_scene.xml

git restore aloha.xml filtered_cartesian_actuators.xml scene.xml

# add a box
cat >mjx_single_cube.xml <<EOF
<mujoco model="aloha with a single cube">
  <size nuserdata="1"/>
  <include file="mjx_scene.xml"/>
  <worldbody>
    <body name="box" pos="0.35 0.2 0.025">
      <freejoint/>
      <geom name="box" type="box" size="0.015 0.02 0.03" condim="3"
        friction="2.5 .03 .003" rgba="0 1 0 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="
      0 -0.96 1.16 0 -0.3 0 0.0084 0.0084
      0 -0.96 1.16 0 -0.3 0 0.0084 0.0084
      0.35 0.2 0.025 1 0 0 0"
      act= "-0.1 0 0 0 0 0 0.03 0.1 0 0 0 0 0 0.03"
      ctrl="-0.1 0 0 0 0 0 0.03 0.1 0 0 0 0 0 0.03"
    />
  </keyframe>
</mujoco>
EOF
```
"""

import argparse
import functools
import os
import sys
import time
from datetime import datetime
from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable

import jax
import jax.numpy as jp
import flax
import mujoco
import numpy as np

import mediapy as media

from brax import base
from brax.io import html
from brax.io import image
from brax.training.agents.ppo import train as ppo
from mujoco.mjx._src import math
from mujoco.mjx._src import io
from mujoco.mjx._src import support

from aloha_env import AlohaBringToTarget

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--benchmark', action='store_true')
arg_parser.add_argument('--render-mj', action='store_true')
arg_parser.add_argument('--use-raytracer', action='store_true')

args = arg_parser.parse_args()


# FIXME, hacky, but need to leave decent chunk of memory for Madrona /
# the batch renderer
def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.55)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


def _measure(fn, *args) -> Tuple[float, float]:
  """Reports jit time and op time for a function."""
  beg = time.time()
  compiled_fn = fn.lower(*args).compile()
  end = time.time()
  jit_time = end - beg

  beg = time.time()
  result = compiled_fn(*args)
  jax.block_until_ready(result)
  end = time.time()
  run_time = end - beg

  return jit_time, run_time


def benchmark(env, nstep, batch_size, unroll_steps=1):
  @jax.pmap
  def init(key):
    key = jax.random.split(key, batch_size // jax.device_count())
    return jax.vmap(env.reset)(key)

  key = jax.random.split(jax.random.key(0), jax.device_count())
  d = init(key)
  jax.block_until_ready(d)

  @jax.pmap
  def unroll(d):
    @jax.vmap
    def step(d, _):
      d = env.step(d, jp.zeros(env.sys.nu))
      return d, None

    d, _ = jax.lax.scan(step, d, None, length=nstep, unroll=unroll_steps)

    return d

  jit_time, run_time = _measure(unroll, d)
  steps = nstep * batch_size
  return jit_time, run_time, steps


if __name__ == '__main__':
  env = AlohaBringToTarget(
      render_batch_size=args.num_worlds,
      gpu_id=args.gpu_id,
      width=args.batch_render_view_width,
      height=args.batch_render_view_height,
      use_rt=args.use_raytracer,
  )
  jit_env_reset = jax.jit(jax.vmap(env.reset))
  jit_env_step = jax.jit(jax.vmap(env.step))

  # rollout the env
  rollout = []
  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)
  state = jit_env_reset(rng=jp.array(key))
  for i in range(args.num_steps):
    act_rng, rng = jax.random.split(rng)
    rollout.append(state)
    ctrl = jax.random.uniform(act_rng, (args.num_worlds, env.sys.nu))
    state = jit_env_step(state, ctrl)

  # render a video for a single env/camera
  for i in range(env.sys.ncam):
    rgbs = np.array([r.info['rgb'][0, i, ..., :3] for r in rollout])
    media.write_video(f'video_madrona_{i}.mp4', rgbs / 255., fps=1.0 / env.dt)

    if args.render_mj:
      camera_name = support.id2name(env.sys.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
      states = [jax.tree_util.tree_map(lambda x: jp.take(x, 0, axis=0), r.pipeline_state)
                for r in rollout]
      rgbs = image.render_array(env.sys, states, 128, 128, camera_name)
      media.write_video(f'video_mujoco_{i}.mp4', rgbs, fps=1.0 / env.dt)

  if args.benchmark:
    jit_time, run_time, steps = benchmark(
        env,
        args.num_steps,
        args.num_worlds,
    )

    print(f"""
Summary for {args.num_worlds} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: { steps / run_time:.0f}
 Total realtime factor: { steps * env.sys.opt.timestep / run_time:.2f} x
 Total time per step: { 1e6 * run_time / steps:.2f} µs""")

