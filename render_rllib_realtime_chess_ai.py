import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_realtime_chess_ai import CNNModelV2, PettingZooChessRtEnv, env_creator

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

        action = DQNAgent.compute_single_action(observation=observation, policy_id=agent)

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
    # Get checkpoint path from checkpoint_path.txt
    checkpoint_path = ""
    read_from = "checkpoint_path.txt"
    with open(read_from, 'r') as f:
        checkpoint_path += f.readline()
        checkpoint_path = checkpoint_path.strip()
        # Exit program if the path is empty
        if checkpoint_path == "":
            print("The checkpoint path doesn't exist.")
            exit(1)

    ray.init()
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
    env_name = "chess_rt"
    register_env(env_name, lambda config: PettingZooChessRtEnv(env_creator("human")))
    
    DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

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