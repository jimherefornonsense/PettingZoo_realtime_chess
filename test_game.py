"""Uses Ray's RLLib to train agents to play Leduc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
from gymnasium.spaces import Box, Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

from pettingzoo.classic import leduc_holdem_v4

torch, nn = try_import_torch()

from typing import Optional
class PettingZooAecEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
    
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
        
        terminated_d["__all__"] = all(self.env.terminations.values())
        truncated_d["__all__"] = all(self.env.truncations.values())

        print(obs_d[agent_id].shape, rew_d, terminated_d, truncated_d, info_d)

        return obs_d, rew_d, terminated_d, truncated_d, info_d
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, options=options)
        return (
            {self.env.agent_selection: self.env.observe(self.env.agent_selection)},
            {},
        )
    
    def render(self):
        return self.env.render()

class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = leduc_holdem_v4.env()
        return env

    env_name = "leduc_holdem_v4"
    register_env(env_name, lambda config: PettingZooAecEnv(env_creator()))

    test_env = PettingZooAecEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {}),
                "player_1": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
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