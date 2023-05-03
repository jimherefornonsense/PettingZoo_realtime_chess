import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_realtime_chess_ai import CNNModelV2, PettingZooChessRtEnv, env_creator

checkpoint_path = "~/path"

alg_name = "DQN"
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

env = env_creator()
env_name = "chess_rt"
register_env(env_name, lambda config: PettingZooChessRtEnv(env_creator()))

ray.init()
DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

reward_sums = {a: 0 for a in env.possible_agents}
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    obs = observation
    reward_sums[agent] += reward

    # Only terminate when one side has no piece left
    if termination:
        break
    # A piece is captured, it's truncated from alive agents
    if truncation:
        env.step(None)
        continue

    print(DQNAgent.get_policy(agent))
    policy = DQNAgent.get_policy(agent)
    batch_obs = {
        "obs": obs,
        "agent_id": agent,
        "infos": info
    }
    batched_action, state_out, info = policy.compute_actions_from_input_dict(
        batch_obs
    )
    single_action = batched_action[0]
    action = single_action

    env.step(action)
    env.render()

print("rewards:")
print(reward_sums)