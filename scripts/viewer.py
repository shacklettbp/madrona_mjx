from sim import Simulator
from madrona_mjx_viz import VisualizerGPUState, Visualizer

import jax
from jax import random, numpy as jp

import argparse

from mjx_policy import policy_init

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)

arg_parser.add_argument('--cpu-sim', action='store_true')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(args.window_width, args.window_height, args.gpu_id)

sim = Simulator(args.gpu_id, args.num_worlds, args.cpu_sim,
                viz_gpu_state.get_gpu_handles())

visualizer = Visualizer(viz_gpu_state, sim.madrona)

@jax.jit
def init_sim_state(rng):
    rng, init_rng = random.split(rng, 2)

    init_mjx_state = sim.reset(init_rng)

    return init_mjx_state, rng

init_mjx_state, rng = init_sim_state(random.key(0))

policy_inference_fn = policy_init(sim.mjx, init_mjx_state)

@jax.jit
def step_fn(carry):
    mjx_state, rng = carry

    rng, step_rng = random.split(rng, 2)
    ctrl = policy_inference_fn(mjx_state, step_rng)
    mjx_state = sim.step(mjx_state, ctrl)

    return mjx_state, rng

visualizer.loop(sim.madrona, step_fn, (init_mjx_state, rng))
