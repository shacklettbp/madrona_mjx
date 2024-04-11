import jax
from jax import random, numpy as jp

import functools

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

def policy_init(env, env_state):
    ppo_network = ppo_networks.make_ppo_networks(
        env_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    params = model.load_params("mjx_brax_quadruped_policy")
    inference_fn = make_inference_fn(params)

    def wrapper(state, rng):
        ctrl, _ = inference_fn(state.obs, rng)
        return ctrl

    return jax.jit(wrapper)
