from typing import Tuple

import gym
import numpy as np

from jaxrl.datasets.dataset import Batch
import jax
import jax.numpy as jnp

from functools import partial
import os
import pickle


class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_seeds: int):
        self.observations = np.empty((num_seeds, capacity, observation_space.shape[-1]),
                                     dtype=observation_space.dtype)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_seeds, capacity, observation_space.shape[-1]),
                                          dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[:, indx],
                     actions=self.actions[:, indx],
                     rewards=self.rewards[:, indx],
                     masks=self.masks[:, indx],
                     next_observations=self.next_observations[:, indx])

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs])

    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.dones_float[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.dones_float[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))
