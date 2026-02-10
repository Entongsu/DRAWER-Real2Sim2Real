#! /usr/bin/env python
import os
import pickle

import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags
import sys

sys.path.append("submodule/rlpd")
import wandb

from scripts.sb3.buffers import ReplayBuffer
import importlib.util
import os
from rlpd.agents import SACLearner
import logging

logging.getLogger("jax").setLevel(logging.ERROR)


def combine_data(offline_data, online_data):
    online_data_dict, offline_data_dict = None, None
    if offline_data is not None:
        offline_data_dict = offline_data._asdict()
    if online_data is not None:
        online_data_dict = online_data._asdict()
    data = {}
    if online_data_dict is not None and offline_data_dict is not None:
        for k, v in offline_data_dict.items():

            data[k] = np.concatenate(
                [offline_data_dict[k], online_data_dict[k]], axis=0)
    elif online_data_dict is not None:
        for k, v in online_data_dict.items():
            data[k] = online_data_dict[k]
    elif offline_data_dict is not None:
        for k, v in offline_data_dict.items():
            data[k] = offline_data_dict[k]
    data["rewards"] = data["rewards"].reshape(-1)
    data["dones"] = data["dones"].reshape(-1)
    return data


class RLPDWrapper:

    def __init__(self, args_cli, env, offline_buffer, seed=100):
        self.args_cli = args_cli
        self.env = env

        self.offline_ratio = self.args_cli.offline_ratio
        self.log_interval = self.args_cli.log_interval
        self.batch_size = self.args_cli.batch_size
        self.max_steps = self.args_cli.max_steps
        self.start_training = self.args_cli.start_training
        self.pretrain_steps = self.args_cli.pretrain_steps
        self.tqdm = True
        self.utd_ratio = self.args_cli.utd_ratio
        self.offline_buffer = offline_buffer
        self.config = self.args_cli.rl_config
        self.seed = seed

        self.init_setting()

    def init_setting(self):
        assert self.offline_ratio >= 0.0 and self.offline_ratio <= 1.0
        from submodule.rlpd.configs import rlpd_config
        self.config = rlpd_config.get_config()
        kwargs = dict(self.config)

        model_cls = kwargs.pop("model_cls")

        self.agent = globals()[model_cls].create(self.seed,
                                                 self.env.observation_space,
                                                 self.env.action_space,
                                                 **kwargs)
        self.num_envs = self.env.env.env.num_envs

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.max_steps,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.env.env.device,
            n_envs=self.env.env.env.num_envs,
        )

    def learn(self, wandb_run):
        for i in tqdm.tqdm(range(0, self.pretrain_steps),
                           smoothing=0.1,
                           disable=not self.tqdm):

            offline_batch = self.offline_buffer.sample(self.batch_size *
                                                       self.utd_ratio)
            batch = {}
            for k, v in offline_batch.items():
                batch[k] = v

            agent, update_info = self.agent.update(batch, self.utd_ratio)

            if i % self.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"offline-training/{k}": v}, step=i)
        observation = self.env.reset()
        reward_buffer = np.zeros(self.num_envs)
        for i in tqdm.tqdm(range(0, self.max_steps + 1),
                           smoothing=0.1,
                           disable=not self.tqdm):
            if i < self.start_training:

                action = np.stack([
                    self.env.action_space.sample()
                    for _ in range(self.num_envs)
                ])
            else:

                action, agent = self.agent.sample_actions(observation)

            next_observation, reward, done, info = self.env.step(action)
            reward_buffer += reward
            if done[0]:
                print("Reward: ", np.mean(reward_buffer))
                reward_buffer = np.zeros(self.num_envs)

            self.replay_buffer.add(
                obs=observation,
                next_obs=next_observation,
                action=action,
                reward=reward,
                done=done,
                infos=info,
            )
            observation = next_observation

            if i >= self.start_training:
                online_batch = self.replay_buffer.sample(
                    int(self.batch_size * self.utd_ratio *
                        (1 - self.offline_ratio)),
                    use_numpy=True)
                offline_batch = self.offline_buffer.sample(
                    int(self.batch_size * self.utd_ratio * self.offline_ratio))

                batch = combine_data(offline_batch, online_batch)

                agent, update_info = agent.update(batch, self.utd_ratio)

                if i % self.log_interval == 0:
                    for k, v in update_info.items():
                        wandb_run.log({f"training/{k}": v},
                                      step=i + self.pretrain_steps)
