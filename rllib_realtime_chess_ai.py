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
from torch import nn
import supersuit as ss

import classic.chess_rt as chess


class PettingZooChessRtEnv(PettingZooEnv):
    def step(self, action):
        self.env.step(action[self.env.agent_selection])
        obs_d = {}
        rew_d = {}
        terminated_d = {}
        truncated_d = {}
        info_d = {}

        while not any(self.env.terminations.values()):
            obs, rew, terminated, truncated, info = self.env.last()
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
        
        if all(self.env.terminations.values()):
            terminated_d = dict(**self.env.terminations)
            truncated_d =  dict(**self.env.truncations)

        return obs_d, rew_d, terminated_d, truncated_d, info_d
    
    def render(self):
        return self.env.render()

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        # Add batch dimension to the input observation
        input_obs = input_dict["obs"].unsqueeze(0)
        model_out = self.model(input_obs.permute(0, 3, 1, 2))
        
        self._value_out = self.value_fn(model_out)

        # Get the policy logits from the policy_fn
        policy_logits = self.policy_fn(model_out)
        
        # Apply the action mask
        action_mask = input_dict["infos"][input_dict["agent_id"]]["action_mask"].unsqueeze(0)  # Add batch dimension
        masked_policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
        
        return masked_policy_logits, state

    def value_function(self):
        return self._value_out.flatten()
    
def env_creator():
    env = chess.env("human")
    # Preprocessing
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 3)
    return env

if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    env_name = "chess_rt"
    register_env(env_name, lambda config: PettingZooChessRtEnv(env_creator()))

    test_env = PettingZooChessRtEnv(env_creator())
    agent_ids = test_env.get_agent_ids()
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    shared_policy = (None, obs_space, act_space, {})

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            dueling=False,
            model={"custom_model": "CNNModelV2"},
        )
        .multi_agent(
            policies={agent_id: shared_policy for agent_id in agent_ids},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )
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
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )