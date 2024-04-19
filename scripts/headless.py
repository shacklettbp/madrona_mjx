import jax
from jax import random, numpy as jp

from madrona_mjx import BatchRenderer
import argparse
from time import time

from mjx_env import MJXEnvAndPolicy

import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)

args = arg_parser.parse_args()

mjx_wrapper = MJXEnvAndPolicy.create(random.key(0), args.num_worlds)

renderer = BatchRenderer.create(
    mjx_wrapper.env, mjx_wrapper.mjx_state, args.gpu_id, args.num_worlds,
    args.batch_render_view_width, args.batch_render_view_height, False)

def step_fn(mjx_wrapper, renderer):
  mjx_wrapper = mjx_wrapper.step()
  renderer, rgb, depth = renderer.render(mjx_wrapper.mjx_state)

  return mjx_wrapper, renderer, rgb, depth

step_fn = jax.jit(step_fn)
step_fn = step_fn.lower(mjx_wrapper, renderer)
step_fn = step_fn.compile()

start = time()

for i in range(args.num_steps):
    mjx_wrapper, renderer, rgb, depth = step_fn(mjx_wrapper, renderer)

end = time()

print("Env Steps Per Second:", "{:.1f}".format(args.num_steps * args.num_worlds / (end - start)))
print("Rendered Views Per Second:", "{:.1f}".format(mjx_wrapper.env.sys.ncam * args.num_steps * args.num_worlds / (end - start)))
