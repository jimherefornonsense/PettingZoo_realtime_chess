import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_realtime_chess_ai import CNNModelV2, PettingZooChessRtEnv, env_creator

CHECKPOINT_PATH = "/Users/zc/ray_results/DQN/DQN_chess_rt_69e8e_00000_0_2023-06-29_19-18-34/checkpoint_000020/"
GAME_COUNTDOWN = 1
TICK_COUNTDOWN = 300

def aec(env, DQNAgent):
    reward_sums = {a: 0 for a in env.possible_agents}
    env.reset()

    tick = 0

    for agent in env.agent_iter():
        print("this is the agent", agent)
        observation, reward, termination, truncation, info = env.last()
        
        # Game over
        if all(env.terminations.values()):
            break
        # A piece is captured, it's terminated/truncated from alive agents
        if termination or truncation:
            reward_sums[agent] = reward
            env.step(None)
            continue

        action = DQNAgent.compute_single_action(observation=observation, info=info, policy_id=agent)

        col = (action//41)%5
        row = action//(5*41)
        print("action index:", action)
        # The coordination is a subjective perspective to the player side, 
        # black side's position is mirrored vertically.
        print("move (x = {}, y = {}, c = {})".format(col, row, action-(row*5+col)*41))
        env.step(action)
        
        print("reward:", reward)

        env.render()
        print("tick:", tick)
        print()
    
        tick += 1
        if tick == TICK_COUNTDOWN:
            break
    
    for alive_agent in env.agents:
        reward_sums[alive_agent] = env._cumulative_rewards[alive_agent]
    print("rewards:")
    print(reward_sums)

# This is for dev purpose.
if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
    env_name = "chess_rt"
    register_env(env_name, lambda config: PettingZooChessRtEnv(env_creator("human")))
    
    DQNAgent = Algorithm.from_checkpoint(CHECKPOINT_PATH)

    # create an environment
    env = env_creator("human")
    game = 0
    
    try:
        while game < GAME_COUNTDOWN:
            aec(env, DQNAgent)
            print("Game", game, "Over")
            game += 1
        
        env.close()
    except KeyboardInterrupt:
        env.close()

    ray.shutdown()