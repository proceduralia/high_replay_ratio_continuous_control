"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from jaxrl.agents.ddpg.actor import update as update_actor
from jaxrl.agents.ddpg.critic import update as update_critic
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None))
def _update(
    actor: Model, target_actor: Model, critic: Model, target_critic: Model, batch: Batch,
    discount: float, tau: float
) -> Tuple[Model, Model, Model, Model, InfoDict]:

    new_critic, critic_info = update_critic(target_actor, critic, target_critic,
                                            batch, discount)
    new_actor, actor_info = update_actor(actor, new_critic, batch)

    new_target_critic = target_update(new_critic, target_critic, tau)
    new_target_actor = target_update(new_actor, target_actor, tau)

    return new_actor, new_target_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
    }


@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'num_updates'))
def _do_multiple_updates(actor: Model, target_actor: Model, critic: Model, target_critic: Model,
                         batches: Batch, discount: float, tau: float,
                         num_updates: int) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        actor, target_actor, critic, target_critic, info = state
        new_actor, new_target_actor, new_critic, new_target_critic, info = _update(
                actor, target_actor, critic, target_critic,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), discount, tau)
        return new_actor, new_target_actor, new_critic, new_target_critic, info
    actor, target_actor, critic, target_critic, info = one_step(0, (actor, target_actor, critic, target_critic, {}))
    return jax.lax.fori_loop(1, num_updates, one_step,
                            (actor, target_actor, critic, target_critic, info))


class DDPGLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.1,
                 num_seeds: int = 1):
        """
        An implementation of [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
        """
        action_dim = actions.shape[-1]
        self.seeds = jnp.arange(seed, seed + num_seeds)
        self.tau = tau
        self.discount = discount
        self.exploration_noise = exploration_noise

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key = jax.random.split(rng, 3)

            actor_def = policies.MSEPolicy(hidden_dims, action_dim)
            actor = Model.create(actor_def,
                                inputs=[actor_key, observations],
                                tx=optax.adam(learning_rate=actor_lr))
            target_actor = Model.create(actor_def,
                                        inputs=[actor_key, observations])

            critic_def = critic_net.Critic(hidden_dims)
            critic = Model.create(critic_def,
                                inputs=[critic_key, observations, actions],
                                tx=optax.adam(learning_rate=critic_lr))
            target_critic = Model.create(
                critic_def, inputs=[critic_key, observations, actions])
            return actor, target_actor, critic, target_critic, rng
        self.init_models = jax.jit(jax.vmap(_init_models))

        self.actor, self.target_actor, self.critic, self.target_critic, self.rng = self.init_models(self.seeds)
        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng,
                                               self.actor.apply_fn,
                                               self.actor.params,
                                               observations,
                                               temperature,
                                               distribution='det')
        self.rng = rng

        actions = np.asarray(actions)
        actions = actions + np.random.normal(
            size=actions.shape) * self.exploration_noise * temperature
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int = 1) -> InfoDict:
        self.step += num_updates
        self.actor, self.target_actor, self.critic, self.target_critic, info = _do_multiple_updates(self.actor, self.target_actor,
                                                                                                    self.critic, self.target_critic,
                                                                                                    batch, self.discount,
                                                                                                    self.tau, num_updates)
        return info

    def reset(self):
        self.step = 1
        self.actor, self.target_actor, self.critic, self.target_critic, self.rng = self.init_models(self.seeds)

    def save_state(self, path: str):
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))
        self.target_critic.save(os.path.join(path, 'target_critic'))
        self.target_actor.save(os.path.join(path, 'target_actor'))
        with open(os.path.join(path, 'step'), 'w') as f:
            f.write(str(self.step))

    def load_state(self, path: str):
        self.actor = self.actor.load(os.path.join(path, 'actor'))
        self.critic = self.critic.load(os.path.join(path, 'critic'))
        self.target_critic = self.target_critic.load(os.path.join(path, 'target_critic'))
        self.target_actor = self.target_actor.load(os.path.join(path, 'target_actor'))
        # Restore the step counter
        with open(os.path.join(path, 'step'), 'r') as f:
            self.step = int(f.read())
