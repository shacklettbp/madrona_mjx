import torch
from madrona_mjx_viz import VisualizerGPUState, Visualizer
from sim import Simulator

import argparse

torch.manual_seed(0)

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
visualizer.loop(sim.madrona, lambda: sim.step())
