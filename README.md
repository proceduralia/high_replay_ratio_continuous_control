# Breaking the Replay Ratio Barrier in Reinforcement Learning for Continuous Control
This repository contains code for high-troughput experimentation with replay ratio scaling based on the ideas from [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier (P. D'Oro*, M. Schwarzer*, E. Nikishin, P.L. Bacon, M. G. Bellemare, A. Courville)](https://openreview.net/forum?id=OpC-9aBBVJe).


The codebase is mostly based on [jaxrl](https://github.com/ikostrikov/jaxrl) and aims to provide the following features:
- **Accessible implementations for off-policy RL**: inherithed from jaxrl, but focused on the online setting (SAC and DDPG)
- **More efficient code for high replay ratio**: additional jit compilation for speed increases at high replay ratios
- **Parallelization over multiple seeds on a single GPU**: multiple seeds are run in parallel by sequentially generating data from each seed's environment but generating new actions and processing updates for all seeds in parallel on the GPU 
- **Off-the-shelf checkpointing**: a simple checkpointing and loading mechanism is provided


Example usage:

`python train_parallel.py --env_name cheetah-run --num_seeds 10 --updates_per_step 32 --max_steps 500000`


The following table reports the approximate running times on a A100 gpu for running 10 seeds per task from the DMC15-500k with replay ratio 32 (a total of 150 seeds, ~16M gradient updates per seed).

| Task | Running Time (approx.) (hrs:min) |
| --- | --- |
| walker-run | 7:15 |
| quadruped-run | 9:30 |
| quadruped-walk | 9:30 |
| reacher-hard | 5:45 |
| humanoid-run | 10:10 |
| humanoid-walk | 10:10 |
| humanoid-stand | 10:10 |
| swimmer-swimmer6 | 7:15 |
| cheetah-run | 7:00 |
| hopper-hop | 6:30 |
| hopper-stand | 6:30 |
| acrobot-swingup | 5:50 |
| pendulum-swingup | 5:20 |
| finger-turn_hard | 6:20 |
| fish-swim | 7:10 |
| TOTAL |  114:25 |

This means less than 5 GPU-days for a rigorous evaluation of your high-replay-ratio learning algorithm. Of course, you can reduce training time by using fewer seeds or tasks during experimentation.

## Checkpointing
Agent and buffer are logged into `FLAGS.save_dir`. The default behavior, which can be overridden, is to load a checkpoint if it is found in the specified directory.

## Results
Results in the `results` directory are generated using the non-parallelized version of the code available in [this repo](https://github.com/evgenii-nikishin/rl_with_resets). They have 5 runs per setting, with 2 million steps per experiment. The names of the directories hint at the different hyperparameters (note: the reset interval is specified in terms of environment steps).

## Citation
If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@inproceedings{
d'oro2023sampleefficient,
title={Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier},
author={Pierluca D'Oro and Max Schwarzer and Evgenii Nikishin and Pierre-Luc Bacon and Marc G Bellemare and Aaron Courville},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=OpC-9aBBVJe}
}
```

