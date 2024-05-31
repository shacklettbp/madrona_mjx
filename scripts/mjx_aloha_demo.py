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
from etils import epath

from jax import core, dtypes
from jax.core import ShapedArray
from jax.lib import xla_client
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters import batching
from jax.interpreters.mlir import ir, dtype_to_ir_type
from jaxlib.hlo_helpers import custom_call

from brax import base
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import html
from brax.io import image
from brax.io import mjcf
from brax.training.agents.ppo import train as ppo
from mujoco.mjx._src import math
from mujoco.mjx._src import io
from mujoco.mjx._src import support

from madrona_mjx.renderer import _setup_jax_primitives
from madrona_mjx._madrona_mjx_batch_renderer import MadronaBatchRenderer
from madrona_mjx._madrona_mjx_batch_renderer.madrona import ExecMode


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)
arg_parser.add_argument('--benchmark', type=bool, required=False, default=False)
arg_parser.add_argument('--render-mj', type=bool, required=False, default=False)

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


def mat_to_quat(mat):
  """Converts 3D rotation matrix to quaternion."""
  # this is a hack to avoid reimplementing logic in mjx.camlight for quats
  mat = mat.flatten()

  q0 = 0.5 * jp.sqrt(1 + mat[0] + mat[4] + mat[8])
  quat0 = jp.array([
      q0,
      0.25 * (mat[7] - mat[5]) / q0,
      0.25 * (mat[2] - mat[6]) / q0,
      0.25 * (mat[3] - mat[1]) / q0
  ])

  q1 = 0.5 * jp.sqrt(1 + mat[0] - mat[4] - mat[8]) 
  quat1 = jp.array([
      q1,
      0.25 * (mat[7] - mat[5]) / q1,
      0.25 * (mat[1] + mat[3]) / q1,
      0.25 * (mat[2] + mat[6]) / q1,
  ])

  q2 = 0.5 * jp.sqrt(1 - mat[0] + mat[4] - mat[8])
  quat2 = jp.array([
      q2,
      0.25 * (mat[2] - mat[6]) / q2,
      0.25 * (mat[1] + mat[3]) / q2,
      0.25 * (mat[5] + mat[7]) / q2,
  ])

  q3 = 0.5 * jp.sqrt(1 - mat[0] - mat[4] + mat[8])
  quat3 = jp.array([
      q3,
      0.25 * (mat[3] - mat[1]) / q3,
      0.25 * (mat[2] + mat[6]) / q3,
      0.25 * (mat[5] + mat[7]) / q3,
  ])

  quat0_cond = (mat[0] + mat[4] + mat[8]) > 0
  quat1_cond = (mat[0] > mat[4]) & (mat[0] > mat[8])
  quat2_cond = mat[4] > mat[8]

  quat = jp.where(quat0_cond, quat0, quat3)
  quat = jp.where(~quat0_cond & quat1_cond, quat1, quat)
  quat = jp.where(~quat0_cond & ~quat1_cond & quat2_cond, quat2, quat)

  return math.normalize(quat)


class Renderer:
  """Wraps MJX Model around MadronaBatchRenderer."""

  def __init__(
      self,
      m,
      gpu_id,
      num_worlds,
      batch_render_view_width,
      batch_render_view_height,
      viz_gpu_hdls=None
  ):
    mesh_verts = m.mesh_vert
    mesh_faces = m.mesh_face
    mesh_vert_offsets = m.mesh_vertadr
    mesh_face_offsets = m.mesh_faceadr
    geom_types = m.geom_type
    # TODO: filter geom groups
    geom_data_ids = m.geom_dataid
    geom_sizes = jax.device_get(m.geom_size)
    # TODO: filter for camera ids
    num_cams = m.ncam

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

    init_fn, render_fn = _setup_jax_primitives(
        self.madrona, num_worlds, geom_sizes.shape[0], num_cams,
        batch_render_view_width, batch_render_view_height)

    self.m = m
    self.init_prim_fn = init_fn
    self.render_prim_fn = render_fn

  def get_geom_quat(self, state):
    to_global = jax.vmap(math.quat_mul)
    geom_quat = to_global(
      state.xquat[self.m.geom_bodyid],
      self.m.geom_quat
    )
    return geom_quat

  def init(self, state):
    geom_quat = self.get_geom_quat(state)
    cam_quat = jax.vmap(mat_to_quat)(state.cam_xmat)

    render_token = jp.array((), jp.bool)

    init_rgb, init_depth, render_token = self.init_prim_fn(
        render_token,
        state.geom_xpos,
        geom_quat,
        state.cam_xpos,
        cam_quat)

    return render_token, init_rgb, init_depth

  def render(self, render_token, state):
    geom_quat = self.get_geom_quat(state)
    cam_quat = jax.vmap(mat_to_quat)(state.cam_xmat)

    render_token = jp.array((), jp.bool)

    rgb, depth, render_token = self.render_prim_fn(render_token,
                          state.geom_xpos,
                          geom_quat,
                          state.cam_xpos,
                          cam_quat)

    return render_token, rgb, depth



def _load_sys(path: str) -> base.System:
  path = epath.Path(path)
  xml = path.read_text()
  assets = {}
  for f in path.parent.glob('*.xml'):
    assets[f.name] = f.read_bytes()
    for f in (path.parent / 'assets').glob('*'):
      assets[f.name] = f.read_bytes()
  model = mujoco.MjModel.from_xml_string(xml, assets)

  # current overrides to get this working:
  model.opt.timestep = 0.005
  model.numeric_data[0] = 10            # raises max_contacts to 10
  model.actuator_dynprm[0:6, 0] = 0.5   # allows actuators to move faster
  model.actuator_dynprm[7:13, 0] = 0.5  # ... leave gripper dynprm as is
  model.opt.iterations = 10             # improves solver stability
  model.opt.ls_iterations = 10          # improves solver stability

  return mjcf.load_model(model)



class AlohaBringToTarget(PipelineEnv):
  """Environment for training aloha to bring an object to target."""

  def __init__(self, render_batch_size: int, gpu_id: int = 0,
               width: int = 128, height: int = 128, **kwargs):
    sys = _load_sys('./mujoco_menagerie/aloha/mjx_single_cube.xml')
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)

    # For rewards
    self._left_gripper_site_index = 3
    self._right_gripper_site_index = 6
    self._box_body_index = sys.mj_model.nbody - 1
    self._box_q_index = 16

    # Init qpos
    self._init_q = jp.array([0.0748744, -0.0298622, 0.869616, 0.0954888,
                             -0.887428, -0.0637624, 0.002, 0.002, -0.0772142,
                             -0.03, 0.869422, -0.100858, -0.887171, 0.0673553,
                             0.002, 0.002, 0, 0, 0.0255684, 1, 0, 0, 0])

    # Action limits and dimension
    self.action_dim = sys.mj_model.actuator_ctrlrange.shape[0]
    self.lowers = sys.mj_model.actuator_ctrlrange[:, 0]
    self.uppers = sys.mj_model.actuator_ctrlrange[:, 1]

    # Madrona renderer
    self.renderer = Renderer(sys, gpu_id, render_batch_size, width, height)

  def reset(self, rng: jax.Array) -> State:
    rng, rng_target, rng_box = jax.random.split(rng, 3)

    target_xy = jax.random.uniform(rng_target, (2,), minval=-0.15, maxval=0.15)
    target_z = jax.random.uniform(rng_target, (1,), minval=0.1, maxval=0.4)
    target_pos = jp.array([0, 0, 0.3])

    box_xy = jax.random.uniform(rng_box, (2,), minval=-0.15, maxval=0.15)
    box_pos = jp.append(box_xy, jp.array(0.02))

    init_q = self._init_q.at[self._box_q_index:self._box_q_index + 3].set(box_pos)
    pipeline_state = self.pipeline_init(init_q, jp.zeros(self.sys.nv))

    info = {'rng': rng, 'target_pos': target_pos}
    obs = self._get_obs(pipeline_state, info)
    reward, done = jp.zeros(2)
    metrics = {'gripper_reward': jp.array(0.0),
               'box_reward': jp.array(0.0),
               'out_of_bounds': jp.array(0.0)}

    render_token, rgb, depth = self.renderer.init(pipeline_state)
    info.update({'render_token': render_token, 'rgb': rgb, 'depth': depth})

    state = State(pipeline_state, obs, reward, done, metrics, info)

    return state

  def step(self, state: State, action: jax.Array) -> State:
    action = jp.clip(action, -1.0, 1.0)
    action = (action + 1) * 0.5  # move from [-1, 1] => [0, 1]
    ctrl = action * (self.uppers - self.lowers) + self.lowers

    data = self.pipeline_step(state.pipeline_state, ctrl)

    render_token, rgb, depth = self.renderer.render(state.info['render_token'], data)
    state.info.update({'render_token': render_token, 'rgb': rgb, 'depth': depth})

    target_pos = state.info['target_pos']
    box_pos = data.xpos[self._box_body_index]

    box_reward = jp.exp(-10 * jp.linalg.norm(target_pos - box_pos))

    left_gripper_pos = data.site_xpos[self._left_gripper_site_index]
    right_gripper_pos = data.site_xpos[self._right_gripper_site_index]
    left_reward = jp.exp(-10 * jp.linalg.norm(box_pos - left_gripper_pos))
    right_reward = jp.exp(-10 * jp.linalg.norm(box_pos - right_gripper_pos))
    gripper_reward = jp.maximum(left_reward, right_reward)

    gripper_open = data.act[6] + data.act[13]

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    reward = 20 * box_reward + gripper_reward + gripper_open
    done = out_of_bounds.astype(float)
    state.metrics.update(
        gripper_reward=gripper_reward,
        box_reward=box_reward,
        out_of_bounds=out_of_bounds.astype(float))
    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_obs(self, data: base.State, info: dict[str, Any]) -> jax.Array:
    # gripper and box positions
    left_gripper_pos = data.site_xpos[self._left_gripper_site_index]
    left_gripper_mat = data.site_xmat[self._left_gripper_site_index].ravel()
    right_gripper_pos = data.site_xpos[self._right_gripper_site_index]
    right_gripper_mat = data.site_xmat[self._right_gripper_site_index].ravel()
    box_pos = data.xpos[self._box_body_index]
    box_mat = data.xmat[self._box_body_index].ravel()
    obs = jp.concatenate([
        data.act,
        left_gripper_pos,
        right_gripper_pos,
        left_gripper_mat,
        right_gripper_mat,
        box_pos,
        box_mat,
        info['target_pos'],
    ])

    return obs


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
      height=args.batch_render_view_height
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

