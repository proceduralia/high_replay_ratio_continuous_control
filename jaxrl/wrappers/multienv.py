import gym
import numpy as np
from gym import spaces


class SequentialMultiEnvWrapper(gym.Env):
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)

    def _reset_idx(self, idx):
        return self.envs[idx].reset()

    def reset_where_done(self, observations, dones):
        for j, done in enumerate(dones):
            if done:
                observations[j], dones[j] = self._reset_idx(j), False
        return observations, dones

    def generate_masks(self, dones, infos):
        masks = []
        for done, info in zip(dones, infos):
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.stack(obs)

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
