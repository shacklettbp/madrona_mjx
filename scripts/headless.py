import jax
from jax import random, numpy as jp

from madrona_mjx import BatchRenderer
import argparse
from time import time

from mjx_env import MJXEnvAndPolicy

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)

args = arg_parser.parse_args()

mjx_wrapper = MJXEnvAndPolicy.create(random.key(0), args.num_worlds)

renderer = BatchRenderer(
    mjx_wrapper.env, mjx_wrapper.mjx_state, args.gpu_id, args.num_worlds,
    args.batch_render_view_width, args.batch_render_view_height, False)

@jax.jit
def mjx_step_wrapper(mjx_wrapper):
  return mjx_wrapper.step()

# Trigger jit
mjx_wrapper = mjx_step_wrapper(mjx_wrapper)

start = time()

for i in range(args.num_steps):
  mjx_wrapper = mjx_step_wrapper(mjx_wrapper)
  renderer.render(mjx_wrapper.mjx_state)

end = time()

print("Env Steps Per Second:", "{:.1f}".format(args.num_steps * args.num_worlds / (end - start)))
print("Rendered Views Per Second:", "{:.1f}".format(mjx_wrapper.env.sys.ncam * args.num_steps * args.num_worlds / (end - start)))
