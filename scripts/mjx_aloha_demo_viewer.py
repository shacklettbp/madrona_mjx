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

# FIXME, hacky, but need to leave decent chunk of memory for Madrona /
# the batch renderer
def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.5)

import jax
import jax.numpy as jp
import flax
import numpy as np

import mediapy as media

from brax.io import html
from brax.io import image
from brax.training.agents.ppo import train as ppo
from mujoco.mjx._src import math
from mujoco.mjx._src import io
from mujoco.mjx._src import support

from aloha_env import AlohaBringToTarget

from madrona_mjx.viz import VisualizerGPUState, Visualizer

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--benchmark', type=bool, required=False, default=False)
arg_parser.add_argument('--add-cam-debug-geo', action='store_true')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(args.window_width, args.window_height, args.gpu_id)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if __name__ == '__main__':
  env = AlohaBringToTarget(
      render_batch_size=args.num_worlds,
      gpu_id=args.gpu_id,
      width=args.batch_render_view_width,
      height=args.batch_render_view_height,
      add_cam_debug_geo=args.add_cam_debug_geo,
      render_viz_gpu_hdls=viz_gpu_state.get_gpu_handles(),
  )
  jit_env_reset = jax.jit(jax.vmap(env.reset))
  jit_env_step = jax.jit(jax.vmap(env.step))

  # rollout the env
  rollout = []
  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)
  state = jit_env_reset(rng=jp.array(key))

  def step_fn(carry):
    rng, state = carry

    act_rng, rng = jax.random.split(rng)
    ctrl = jax.random.uniform(act_rng, (args.num_worlds, env.sys.nu))
    state = jit_env_step(state, ctrl)

    return rng, state
    
  visualizer = Visualizer(viz_gpu_state, env.renderer.madrona)
  visualizer.loop(env.renderer.madrona, step_fn, (rng, state))
