"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))
def _update(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float, target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            soft_critic=True)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'num_updates'))
def _do_multiple_updates(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                         temp: Model, batches: Batch, discount: float, tau: float,
                         target_entropy: float, step, num_updates: int) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
                rng, actor, critic, target_critic, temp,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), discount, tau, target_entropy)
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info
    step, rng, actor, critic, target_critic, temp, info = one_step(0, (step, rng, actor, critic, target_critic, temp, {}))
    return jax.lax.fori_loop(1, num_updates, one_step,
                             (step, rng, actor, critic, target_critic, temp, info))


class SACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 num_seeds: int = 5) -> None:
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.seeds = jnp.arange(seed, seed + num_seeds)
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.discount = discount

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
            actor = Model.create(actor_def,
                                 inputs=[actor_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))

            critic_def = critic_net.DoubleCritic(hidden_dims)
            critic = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions],
                                  tx=optax.adam(learning_rate=critic_lr))
            target_critic = Model.create(
                critic_def, inputs=[critic_key, observations, actions])

            temp = Model.create(temperature.Temperature(init_temperature),
                                inputs=[temp_key],
                                tx=optax.adam(learning_rate=temp_lr))
            return actor, critic, target_critic, temp, rng
        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        self.step = 1

    def save_state(self, path: str):
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))
        self.target_critic.save(os.path.join(path, 'target_critic'))
        self.temp.save(os.path.join(path, 'temp'))
        with open(os.path.join(path, 'step'), 'w') as f:
            f.write(str(self.step))

    def load_state(self, path: str):
        self.actor = self.actor.load(os.path.join(path, 'actor'))
        self.critic = self.critic.load(os.path.join(path, 'critic'))
        self.target_critic = self.target_critic.load(os.path.join(path, 'target_critic'))
        self.temp = self.temp.load(os.path.join(path, 'temp'))
        # Restore the step counter
        with open(os.path.join(path, 'step'), 'r') as f:
            self.step = int(f.read())


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int = 1) -> InfoDict:
        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step, num_updates)

        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        return info

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
