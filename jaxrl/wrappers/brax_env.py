
# Modified Brax wrapper from https://github.com/google/brax/blob/main/brax/envs/wrappers.py
from typing import ClassVar
from brax import jumpy as jp
from gym import core, spaces
import jax
from brax import envs
from brax.envs.env import State
from brax.envs import env as brax_env


class BraxEnv(core.Env):
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env_name: str, seed: int = 0):
        self.env_name = env_name
        self._env = envs.create(env_name=env_name)
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.sys.config.dt,
        }
        self.seed(seed)
        self._state = None

        obs_high = jp.inf * jp.ones(self._env.observation_size, dtype="float32")
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype="float32")

        action_high = jp.ones(self._env.action_size, dtype="float32")
        self.action_space = spaces.Box(-action_high, action_high, dtype="float32")

        def reset(key):
            key1, key2 = jp.random_split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step)

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        # We return device arrays for pytorch users.
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        # We return device arrays for pytorch users.
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def get_state(self) -> State:
        return self._state

    def set_state(self, state: State):
        self._state = state

    def render(self, mode="human"):
        # pylint:disable=g-import-not-at-top
        from brax.io import image

        if mode == "rgb_array":
            sys, qp = self._env.sys, self._state.qp
            return image.render_array(sys, qp, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception


class BraxGymWrapper(core.Env):
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: brax_env.Wrapper, seed: int, num_seeds: int):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.sys.config.dt,
        }
        self._state = None

        def create_PRNGKey(seed):
            return jax.random.PRNGKey(seed)
        self._key = jax.vmap(create_PRNGKey)(jp.arange(seed, seed + num_seeds))

        obs_high = jp.inf * jp.ones((num_seeds, self._env.observation_size), dtype="float32")
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype="float32")

        action_high = jp.ones((num_seeds, self._env.action_size), dtype="float32")
        self.action_space = spaces.Box(-action_high, action_high, dtype="float32")

        def reset(key):
            keys = jax.vmap(jp.random_split)(key)
            key1, key2 = keys[:, 0, :], keys[:, 1, :]
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step)

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        return obs

    def reset_where_done(self, _, __):
        # This function checks if at least one of the environments is done, and if so, computes a new state for all of them
        if 'steps' in self._state.info:
            steps = self._state.info['steps']
            steps = jp.where(self._state.done, jp.zeros_like(steps), steps)
            self._state.info.update(steps=steps)

        if jp.any(self._state.done):
            def where_done(x, y):
                done = jp.reshape(self._state.done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
                return jp.where(done, x, y)
            new_state, _, self._key = self._reset(self._key)
            qp = jp.tree_map(where_done, new_state.qp, self._state.qp)
            obs = where_done(new_state.obs, self._state.obs)
            self._state = self._state.replace(qp=qp, obs=obs)
            self._state = self._state.replace(done=jp.zeros_like(self._state.done))
        return self._state.obs, self._state.done

    def generate_masks(self, dones, infos):
        return ~dones.astype('bool') | infos['truncation'].astype('bool')

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        return obs, reward, done, info

    def render(self, mode="human"):
        raise NotImplementedError