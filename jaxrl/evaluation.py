from typing import Dict
import numpy as np
import gym

from jaxrl.networks.common import Model
import jax.numpy as jnp
from jax.random import KeyArray
from typing import Callable
import jax
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, 0, None))
def evaluate_brax(actor: Model, eval_step_fn: Callable, eval_reset_fn: Callable,
                  num_episodes: int, rng: KeyArray, episode_length: int = 1000) -> Dict[str, float]:
    rng, key = jax.random.split(rng)
    state = eval_reset_fn(rng=key)
    def one_step(i, state_ret):
        state, ret = state_ret
        dist = actor.apply({'params': actor.params}, state.obs, 0.0)
        try:
            action = dist.sample(seed=key)
        except AttributeError:
            action = dist
        next_state = eval_step_fn(state, action)
        ret = ret + next_state.reward
        return (next_state, ret)
    ret = jnp.zeros(num_episodes)
    last_state, ret = jax.lax.fori_loop(0, episode_length, one_step, (state, ret))
    eval_metrics = last_state.info['eval_metrics']
    avg_episode_length = (
          eval_metrics.completed_episodes_steps /
          eval_metrics.completed_episodes)
    metrics = dict(
          dict({
              f'eval_episode_{name}': value / eval_metrics.completed_episodes
              for name, value in eval_metrics.completed_episodes_metrics.items()
          }),
          **dict({
              'eval_completed_episodes': eval_metrics.completed_episodes,
              'eval_avg_episode_length': avg_episode_length
          }))
    metrics['return'] = metrics["eval_episode_reward"]
    return metrics


def evaluate(agent, env: gym.Env, num_episodes: int, episode_length: int) -> Dict[str, float]:
    if 'brax' in str(type(env)).lower():
        return evaluate_brax(agent.actor, env.step, env.reset, num_episodes, agent.rng, episode_length)
    else:
        n_seeds = env.num_envs
        returns = []
        for _ in range(num_episodes):
            observations, dones = env.reset(), np.array([False] * n_seeds)
            rets, length = np.zeros(n_seeds), 0
            while not dones.all():
                actions = agent.sample_actions(observations, temperature=0.0)
                prev_dones = dones
                observations, rewards, dones, infos = env.step(actions)
                rets += rewards * (1 - prev_dones)
                length += 1
                if length >= episode_length:
                    break
            returns.append(rets)
        return {'return': np.array(returns).mean(axis=0)}
