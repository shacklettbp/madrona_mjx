"""Custom wrappers that extend Brax wrappers"""

import functools
from typing import Callable, Optional, Tuple

from brax.base import System
from brax.envs.base import Env
from brax.envs.base import State
from brax.envs.base import Wrapper
from etils import epath
import jax
from jax import numpy as jp
import mujoco


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


def _identity_randomization_fn(
    sys: System, num_worlds: int
) -> Tuple[System, System]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })

  sys = sys.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(sys.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(sys.geom_matid, 0), num_worlds, axis=0
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(sys.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(sys.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(sys.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(sys.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(sys.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(sys.light_cutoff, 0), num_worlds, axis=0
      ),
  })

  return sys, in_axes


class MadronaWrapper(Wrapper):
  """Wrapper to Vmap an environment that uses the Madrona BatchRenderer.

  Madrona expects certain MjModel axes to be batched so that the buffers can
  be copied to the GPU. Therefore we need to dummy batch the model to create
  the correct sized buffers for those not using randomization functions,
  and for those using randomization we ensure the correct axes are batched.

  Use this instead of the Brax VmapWrapper and DomainRandimzationWrapper."""

  def __init__(
      self,
      env: Env,
      num_worlds,
      randomization_fn: Optional[
          Callable[[System], Tuple[System, System]]
      ] = None,
  ):
    super().__init__(env)
    self.num_worlds = num_worlds
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_randomization_fn, num_worlds=num_worlds
      )

    self._sys_v, self._in_axes = randomization_fn(self.sys)
    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def _env_fn(self, sys: System) -> Env:
    env = self.env
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._sys_v, state, action
    )
    return res
