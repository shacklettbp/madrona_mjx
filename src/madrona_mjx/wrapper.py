"""Custom wrappers that extend Brax wrappers"""

from typing import Callable, Optional, Tuple
from etils import epath

import jax
from jax import numpy as jp
import mujoco
from brax.base import System
from brax.envs.base import Env, State, Wrapper


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


def dummy_tile(sys: System, num_worlds: int) -> Tuple[System, System]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
  })

  sys = sys.tree_replace({
    'geom_rgba': jp.repeat(
      jp.expand_dims(sys.geom_rgba, 0), num_worlds, axis=0),
    'geom_matid': jp.repeat(
      jp.expand_dims(sys.geom_matid, 0), num_worlds, axis=0),
    'geom_size': jp.repeat(
      jp.expand_dims(sys.geom_size, 0), num_worlds, axis=0),
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
      randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
  ):
    super().__init__(env)
    self.num_worlds = num_worlds
    if randomization_fn is not None:
      self._sys_v, self._in_axes = randomization_fn(self.sys)
      # For user made DR functions, ensure that the output model includes the
      # needed in_axes for madrona initialization.
      assert self._in_axes is not None
      assert self._in_axes.geom_rgba is not None
      assert self._in_axes.geom_matid is not None
      assert self._in_axes.geom_size is not None
      
      # Check shapes in model are consistent with num_worlds
      assert self._sys_v.geom_rgba.shape[0] == num_worlds
      assert self._sys_v.geom_matid.shape[0] == num_worlds
      assert self._sys_v.geom_size.shape[0] == num_worlds
    else:
      self._sys_v, self._in_axes = dummy_tile(self.sys, self.num_worlds)

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