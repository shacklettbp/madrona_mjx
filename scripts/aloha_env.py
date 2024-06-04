import jax
import jax.numpy as jp
import mujoco

from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable

from brax import base
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import mjcf

from etils import epath

from madrona_mjx.renderer import BatchRenderer


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
               width: int = 128, height: int = 128,
               add_cam_debug_geo: bool = False, 
               render_viz_gpu_hdls = None, **kwargs):
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
    self.renderer = BatchRenderer(sys, gpu_id, render_batch_size, 
                                  width, height, add_cam_debug_geo,
                                  render_viz_gpu_hdls)

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

