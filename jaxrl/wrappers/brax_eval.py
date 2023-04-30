from brax.envs import env as brax_env
from brax import jumpy as jp
import jax
import flax
from typing import Dict, Optional

def zero_metrics_like(metrics_tree):
    eval_metrics = EvalMetrics(
        current_episode_metrics=jax.tree_map(jp.zeros_like,
                                             metrics_tree),
        completed_episodes_metrics=jax.tree_map(
            lambda x: jp.zeros_like(jp.sum(x)), metrics_tree),
        completed_episodes=jp.zeros(()),
        completed_episodes_steps=jp.zeros(()),
        discounted_current_episode_metrics=jax.tree_map(jp.zeros_like,
                                            metrics_tree),
        discounted_completed_episodes_metrics=jax.tree_map(
            lambda x: jp.zeros_like(jp.sum(x)), metrics_tree),
        current_discount=jax.tree_map(jp.ones_like,
                                      metrics_tree))
    return eval_metrics


@flax.struct.dataclass
class EvalMetrics:
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray
    discounted_current_episode_metrics: Dict[str, jp.ndarray]
    discounted_completed_episodes_metrics: Dict[str, jp.ndarray]
    current_discount: Dict[str, jp.ndarray]


class BraxEvalWrapper(brax_env.Wrapper):
    """Brax env with eval metrics."""
    def reset(self, rng: jp.ndarray) -> brax_env.State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        reset_state.info['eval_metrics'] = zero_metrics_like(reset_state.metrics)
        return reset_state

    def step(self, state: brax_env.State, action: jp.ndarray, discount: float = 0.99, reset_state: Optional[brax_env.State] = None) -> brax_env.State:
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
             f'Incorrect type for state_metrics: {type(state_metrics)}')
        del state.info['eval_metrics']
        if reset_state is not None:
            nstate = self.env.step(state, action, reset_state=reset_state)
        else:
            nstate = self.env.step(state, action)
        nstate.metrics['reward'] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info['steps'] * nstate.done)
        current_episode_metrics = jax.tree_multimap(
            lambda a, b: a + b, state_metrics.current_episode_metrics,
            nstate.metrics)
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_multimap(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics, current_episode_metrics)
        current_episode_metrics = jax.tree_multimap(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics, nstate.metrics)

        # Update discount
        current_discount = jax.tree_multimap(lambda a:  a * discount, state_metrics.current_discount)
        current_discount = jax.tree_multimap(lambda a:  a * (1 - nstate.done) + 1 * nstate.done, current_discount)

        # Discounted metrics
        discounted_current_episode_metrics = jax.tree_multimap(
            lambda a, b, gamma: a + b*gamma, state_metrics.discounted_current_episode_metrics,
            nstate.metrics, current_discount)
        discounted_completed_episodes_metrics = jax.tree_multimap(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.discounted_completed_episodes_metrics, discounted_current_episode_metrics)
        discounted_current_episode_metrics = jax.tree_multimap(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            discounted_current_episode_metrics, nstate.metrics)

        eval_metrics = EvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
            discounted_current_episode_metrics=discounted_current_episode_metrics,
            discounted_completed_episodes_metrics=discounted_completed_episodes_metrics,
            current_discount=current_discount)
        nstate.info['eval_metrics'] = eval_metrics
        return nstate


class AutoResetWrapper(brax_env.Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    state = self.env.reset(rng)
    state.info['first_qp'] = state.qp
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: brax_env.State, action: jp.ndarray, reset_state: Optional[brax_env.State] = None) -> brax_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    if reset_state is not None:
        qp = jp.tree_map(where_done, reset_state.qp, state.qp)
        obs = where_done(reset_state.obs, state.obs)
    else:
        qp = jp.tree_map(where_done, state.info['first_qp'], state.qp)
        obs = where_done(state.info['first_obs'], state.obs)
    return state.replace(qp=qp, obs=obs)