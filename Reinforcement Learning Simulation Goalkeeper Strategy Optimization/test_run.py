try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from learners  import MeanQLearner, FixedRateQLearner, PreferenceLearner
from environment import get_goal_keeper

GOAL_KEEPERS = ["biased",]

def run_match(agent, goal_keeper, kicks_per_match=50):
    total_reward = 0
    for _ in range(kicks_per_match):
        action = agent.choose_action()
        _, reward = goal_keeper.step(action)
        agent.update(action, reward)
        total_reward += reward
    return total_reward

def main(matches=10000, kicks_per_match=50):
    
    # example testing of a specific agent:
    for goal_keeper_type in GOAL_KEEPERS:
        results = []
        for match_num in tqdm(range(matches), desc=f"Testing {goal_keeper_type} Goal Keeper") if tqdm else range(matches):
            goal_keeper = get_goal_keeper(goal_keeper_type)
            agent = MeanQLearner(epsilon=0.1, seed=None)
            result = run_match(agent, goal_keeper, kicks_per_match=kicks_per_match)
            results.append(result)
        avg_reward = sum(results) / matches
        print(f"Goal Keeper: {goal_keeper_type}, MeanQLearner Average Reward over {matches} matches: {avg_reward:.3f}")

if __name__ == "__main__":
    main(matches=1000, kicks_per_match=50)
