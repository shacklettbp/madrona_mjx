"""A cartpole environment."""
import os
import functools
import time
import numpy as np
import argparse
from pathlib import Path
import jax
import jax.numpy as jp
import mujoco
from flax.core import FrozenDict

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.envs import training

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks_vision as ppo_networks_vision

from ml_collections import config_dict

from madrona_mjx import BatchRenderer
from madrona_mjx.wrapper import MadronaWrapper


def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.1)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

_XML_DIR = Path(__file__).resolve().parent.parent / "data"
_XML = _XML_DIR / "cartpole.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
    )


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=1,
        render_width=64,
        render_height=64,
        enabled_geom_groups=np.array([0, 1, 2]),
        use_rasterizer=False,
    )


class Cartpole(PipelineEnv):
  def __init__(
        self,
        vision: bool = False,
        config: config_dict.ConfigDict = default_config(),
        vision_config: config_dict.ConfigDict = default_vision_config()):

    mj_model = mujoco.MjModel.from_xml_path(str(_XML))
    sys = mjcf.load_model(mj_model)
    self._env_cfg = config
    self._vision = vision
    self._vision_cfg = vision_config
    n_frames = config.ctrl_dt / config.sim_dt
    backend = 'mjx'

    super().__init__(sys=sys, n_frames=n_frames, backend=backend)

    if vision:
      self.renderer = BatchRenderer(
          sys,
          vision_config.gpu_id,
          vision_config.render_batch_size,
          vision_config.render_width,
          vision_config.render_height,
          vision_config.enabled_geom_groups,
          False,
          vision_config.use_rasterizer,
          None)

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
    )
    qd = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
    )
    pipeline_state = self.pipeline_init(q, qd)

    info = {}

    if self._vision:
      render_token, rgb, depth = self.renderer.init(pipeline_state, self.sys)
      obs = FrozenDict({"pixels/rgb": jp.asarray(rgb[0][:, :, :3], dtype=jp.float32) / 255.0})
      info.update({'render_token': render_token,})
    else:
      obs = self._get_obs(pipeline_state, info)

    reward, done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

    # Scale action from [-1,1] to actuator limits
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    pipeline_state = self.pipeline_step(state.pipeline_state, action)

    if self._vision:
      _, rgb, depth = self.renderer.render(state.info['render_token'], pipeline_state)
      obs = FrozenDict({"pixels/rgb": jp.asarray(rgb[0][:, :, :3], dtype=jp.float32) / 255.0})
    else:
      obs = self._get_obs(pipeline_state, action)

    reward = 1.0
    done = jp.where(jp.abs(pipeline_state.q[1]) > 0.2, 1.0, 0.0)
    return state.replace(
          pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=state.info)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--num-envs', type=int, required=True)
  arg_parser.add_argument('--num-steps', type=int, default=5_000_000)
  args = arg_parser.parse_args()

  env_config = default_config()
  env_vision_config = default_vision_config()
  env_vision_config.render_batch_size = args.num_envs
  env = Cartpole(vision=True, config=env_config, vision_config=env_vision_config)

  make_networks_factory = functools.partial(
      ppo_networks_vision.make_ppo_networks_vision,
      policy_hidden_layer_sizes=(128, 128, 128),
      value_hidden_layer_sizes=(128, 128, 128),
  )

  train_fn = functools.partial(
      ppo.train,
      num_timesteps=1_000_000,
      num_evals=5,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=10,
      num_minibatches=8,
      num_updates_per_batch=8,
      discounting=0.95,
      learning_rate=3e-4,
      entropy_cost=1e-3,
      num_envs=args.num_envs,
      num_eval_envs=args.num_envs,
      batch_size=256,
      network_factory=make_networks_factory,
      num_resets_per_eval=1,
      seed=0,
      wrap_env=False
  )

  env = MadronaWrapper(env, args.num_envs)
  env = training.EpisodeWrapper(env, episode_length=env_config.episode_length, action_repeat=1)
  env = training.AutoResetWrapper(env)

  def progress(num_steps, metrics):
    print(f'step: {num_steps}, reward: {metrics["eval/episode_reward"]}')

  print("Starting training...")
  start = time.time()
  make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)
  end = time.time()
  train_time = end - start

  print(
    f"""
    Summary for {args.env} gpu training
    Total simulation time: {train_time:.2f} s
    Total training wall time: {metrics['training/walltime']} s
    Total eval wall time: {metrics['eval/walltime']} s
    Total time per step: { 1e6 * train_time / args.num_steps:.2f} µs""")
