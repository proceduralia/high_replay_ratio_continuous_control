from typing import Optional

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers
from jaxrl.wrappers import VideoRecorder, SequentialMultiEnvWrapper, BraxEvalWrapper, AutoResetWrapper, BraxGymWrapper
from brax.envs import _envs as brax_envs
from brax import envs
import jax


def make_one_env(env_name: str,
                 seed: int,
                 save_folder: Optional[str] = None,
                 add_episode_monitor: bool = True,
                 action_repeat: int = 1,
                 frame_stack: int = 1,
                 from_pixels: bool = False,
                 pixels_only: bool = True,
                 image_size: int = 84,
                 sticky: bool = False,
                 gray_scale: bool = False,
                 flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def create_brax_env(env_name: str,
                    episode_length: int = 1000,
                    action_repeat: int = 1,
                    auto_reset: bool = True,
                    batch_size: Optional[int] = None,
                    eval_metrics: bool = False,
                    seed: int = 0,
                    **kwargs) -> envs.env.Env:
    """Creates an Env with a specified brax system."""
    env = envs._envs[env_name](**kwargs)
    if episode_length is not None:
        env = envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        if eval_metrics:
            env = envs.wrappers.VectorWrapper(env, batch_size)
        else:
            env = envs.wrappers.VmapWrapper(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    # ATTENTION: BraxEvalWrapper requires AutoResetWrapper to be the one right before it
    if eval_metrics:
        env = BraxEvalWrapper(env)
    else:
        env = BraxGymWrapper(env, seed=seed, num_seeds=batch_size)
    return env


def make_env(env_name: str,
             seed: int,
             eval_episodes: Optional[int] = None,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             eval_episode_length: int = 1000,
             num_envs: Optional[int] = None) -> gym.Env:
    if env_name in brax_envs:
        if eval_episodes:
            env = create_brax_env(env_name=env_name, episode_length=eval_episode_length, eval_metrics=True, batch_size=eval_episodes)
            env.step = jax.jit(env.step)
            env.reset = jax.jit(env.reset)
            return env
        else:
            env = create_brax_env(env_name=env_name, episode_length=eval_episode_length, batch_size=num_envs,
                                  auto_reset=False, eval_metrics=False, seed=seed)
            return env
    else:
        if num_envs is None:
            return make_one_env(env_name, seed, save_folder, add_episode_monitor, action_repeat, frame_stack, from_pixels, pixels_only, image_size, sticky, gray_scale, flatten)
        else:
            env_fn_list = [lambda: make_one_env(env_name, seed+i, save_folder, add_episode_monitor, action_repeat, frame_stack,
                                                from_pixels, pixels_only, image_size, sticky, gray_scale, flatten)
                           for i in range(num_envs)]
            return SequentialMultiEnvWrapper(env_fn_list)
