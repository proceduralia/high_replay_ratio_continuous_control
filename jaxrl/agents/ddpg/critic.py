from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params


def update(target_actor: Model, critic: Model, target_critic: Model, batch: Batch,
           discount: float) -> Tuple[Model, InfoDict]:
    next_actions = target_actor(batch.next_observations)
    next_q = target_critic(batch.next_observations, next_actions)

    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q = critic.apply({'params': critic_params}, batch.observations, batch.actions)
        critic_loss = ((q - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': q.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
