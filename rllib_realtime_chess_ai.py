#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:03:57 2023

@author: zc
"""

import os

import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX
import torch
from torch import nn
import supersuit as ss

import classic.chess_rt as chess

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from typing import Optional
class PettingZooChessRtEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.counter = 0

    def step(self, action):
        # eprint("turn:", self.env.agent_selection)
        self.env.step(action[self.env.agent_selection])
        obs_d = {}
        rew_d = {}
        terminated_d = {}
        truncated_d = {}
        info_d = {}

        while not all(self.env.terminations.values()):
            obs, rew, terminated, truncated, info = self.env.last()
            # eprint("now:", self.env.agent_selection)
            # eprint("terminated:",terminated)
            agent_id = self.env.agent_selection
            obs_d[agent_id] = obs
            rew_d[agent_id] = rew
            terminated_d[agent_id] = terminated
            truncated_d[agent_id] = truncated
            info_d[agent_id] = info
            if (
                self.env.terminations[self.env.agent_selection]
                or self.env.truncations[self.env.agent_selection]
            ):
                self.env.step(None)
            else:
                break
        
        terminated_d["__all__"] = all(self.env.terminations.values())
        truncated_d["__all__"] = all(self.env.truncations.values())


        self.counter += 1
        # eprint("counter: ", self.counter)
        # eprint("")

        return obs_d, rew_d, terminated_d, truncated_d, info_d
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        info = self.env.reset(seed=seed, options=options)
        obs = self.env.observe(self.env.agent_selection)
        return (
            {self.env.agent_selection: obs},
            info or {},
        )
    
    def render(self):
        return self.env.render()

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, act_space.n, *args, **kwargs)
        nn.Module.__init__(self)

        # Define the CNN layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # The size here is an example, please adjust according to your output from the Conv2D layers
            nn.ReLU()
        )

        self.policy_fn = nn.Linear(512, act_space.n) #1026
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        input_obs = input_dict["obs"]["observation"]
        model_out = self.model(input_obs.permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)

        # Get the policy logits from the policy_fn
        policy_logits = self.policy_fn(model_out)

        # Apply the action mask
        action_mask = input_dict["obs"]["action_mask"]
        if action_mask.any():
            # action_mask_tensor = torch.from_numpy(action_mask)
            inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
            policy_logits = inf_mask + policy_logits
        
        return policy_logits, state

    def value_function(self):
        return self._value_out.flatten()
    
def env_creator(render_mode):
    env = chess.env(render_mode=render_mode)
    # Preprocessing [400, 400, 3]
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env

if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    env_name = "chess_rt"
    register_env(env_name, lambda config: PettingZooChessRtEnv(env_creator("rgb_array")))

    test_env = PettingZooChessRtEnv(env_creator("rgb_array"))
    agent_ids = test_env.get_agent_ids()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=20, preprocessor_pref=None)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={
                "custom_model": "CNNModelV2",
            },
        )
        .multi_agent(
            policies={agent_id: (None, obs_space, act_space, {}) for agent_id in agent_ids},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        # .debugging(
        #     log_level="DEBUG"
        # )
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 40000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )

    ray.shutdown()