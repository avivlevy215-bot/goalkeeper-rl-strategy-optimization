import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environment import get_goal_keeper
from environment import Biased_Goal_Keeper
from environment import GoalKeeper

class MeanQLearner:
    """Q-learner that tracks the sample-mean value for each action."""

    def __init__(self, actions=("left", "right"), epsilon=0.1, seed=None):
        self.actions = list(actions)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.q = {action: 0.0 for action in self.actions}
        self.counts = {action: 0 for action in self.actions}

    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        max_q = max(self.q.values())
        best_actions = [a for a, v in self.q.items() if v == max_q]
        return self.rng.choice(best_actions)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.q[action] += (reward - self.q[action]) / n


class FixedRateQLearner:
    def __init__(self, actions=("left", "right"), eta=0.1, epsilon=0.1, seed=None):
        self.actions = list(actions)
        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.q = {action: 0.0 for action in self.actions}


    def choose_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        max_q = max(self.q.values())
        best_actions = [a for a, v in self.q.items() if v == max_q]
        return self.rng.choice(best_actions)


    def update(self, action, reward):
        self.q[action] += self.eta * (reward - self.q[action])


class PreferenceLearner:
    def __init__(self, actions=("left", "right"), eta=0.5, seed=None):
        self.actions = list(actions)
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        self.h = {action: 0.0 for action in self.actions}  # Preferences
        self.avg_reward = 0.0  # Baseline
        self.count = 0

    def choose_action(self):
        exp_h = {a: np.exp(v) for a, v in self.h.items()}
        sum_exp = sum(exp_h.values())
        probs = [exp_h[a] / sum_exp for a in self.actions]
        return self.rng.choice(self.actions, p=probs)

    def update(self, action, reward):
        self.count += 1
        self.avg_reward += (reward - self.avg_reward) / self.count

        exp_h = {a: np.exp(v) for a, v in self.h.items()}
        sum_exp = sum(exp_h.values())

        for a in self.actions:
            pi_a = exp_h[a] / sum_exp
            if a == action:
                self.h[a] += self.eta * (reward - self.avg_reward) * (1 - pi_a)
            else:
                self.h[a] += self.eta * (reward - self.avg_reward) * (-pi_a)

# bounos_class
class Adaptive_Goal_Keeper(GoalKeeper):
    """
    Adaptive goalkeeper that models the kicker's action distribution.
    Uses Thompson Sampling + forgetting to be robust vs low-epsilon learners.
    """

    def __init__(self, seed=None, alpha0=1.0, beta0=1.0, decay=0.98):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.alpha = float(alpha0)
        self.beta = float(beta0)
        self.decay = float(decay)

    def predecide_goal_keeper_action(self):
        p_left = self.rng.beta(self.alpha, self.beta)
        return "left" if p_left >= 0.5 else "right"

    def step(self, kicker_action):
        decision, reward = super().step(kicker_action)

        self.alpha = self.alpha0 + self.decay * (self.alpha - self.alpha0)
        self.beta = self.beta0 + self.decay * (self.beta - self.beta0)

        if kicker_action == "left":
            self.alpha += 1.0
        else:
            self.beta += 1.0

        return decision, reward



def run_match(agent_factory, goalkeeper_factory, kicks_per_match=50, num_matches=1000):
    rewards = []
    for match_i in range(num_matches):
        agent = agent_factory(match_i)
        keeper = goalkeeper_factory(match_i)

        total_reward = 0
        for _ in range(kicks_per_match):
            action = agent.choose_action()
            _, reward = keeper.step(action)
            agent.update(action, reward)
            total_reward += reward

        rewards.append(total_reward)

    return float(np.mean(rewards))

def run_match_rewards(agent_factory, goalkeeper_factory, kicks_per_match=50, num_matches=1000):
    rewards = []
    for match_i in range(num_matches):
        agent = agent_factory(match_i)
        keeper = goalkeeper_factory(match_i)

        total_reward = 0
        for _ in range(kicks_per_match):
            action = agent.choose_action()
            _, reward = keeper.step(action)
            agent.update(action, reward)
            total_reward += reward

        rewards.append(total_reward)

    return np.array(rewards, dtype=float)
def unbiased_evaluation(agent_factory, keeper_factory,
                        num_matches=10000,
                        kicks_per_match=50):

    rewards = run_match_rewards(
        agent_factory=lambda match_i: agent_factory(seed=None),
        goalkeeper_factory=lambda match_i: keeper_factory(seed=None),
        num_matches=num_matches,
        kicks_per_match=kicks_per_match,
    )

    return rewards.mean(), rewards.std()

def main():
    num_matches = 1000

    # --- MeanQLearner: tune epsilon ---
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2,0.4]
    result_e = []

    for eps in epsilons:
        avg_reward = run_match(
            agent_factory=lambda i: MeanQLearner(epsilon=eps,seed=None),
            goalkeeper_factory=lambda i: get_goal_keeper("biased",seed=None),
            num_matches=num_matches,
            kicks_per_match=50,
        )
        result_e.append(avg_reward)

    best_eps = epsilons[int(np.argmax(result_e))]
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, result_e, marker="o", label="Mean Reward")
    plt.plot(best_eps, max(result_e), "r*", markersize=15, label=f"Best (eps={best_eps})")
    plt.title("MeanQLearner: Epsilon Tuning")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

 # --- PreferenceLearner: tune eta ---
    etas_pref = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]
    pref_results = []
    for eta in etas_pref:
        avg_reward = run_match(
            agent_factory=lambda i: PreferenceLearner(eta=eta, seed=None),
            goalkeeper_factory=lambda i: get_goal_keeper("biased", seed=None),
            num_matches=num_matches,
            kicks_per_match=50,
        )
        pref_results.append(avg_reward)

    best_eta_pref = etas_pref[int(np.argmax(pref_results))]
    plt.figure(figsize=(8, 5))
    plt.plot(etas_pref, pref_results, marker="o", label="Mean Reward")
    plt.plot(best_eta_pref, max(pref_results), "r*", markersize=15, label=f"Best (eta={best_eta_pref})")
    plt.title("PreferenceLearner: Eta Tuning")
    plt.xlabel("Eta")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- FixedRateQLearner: grid search eta & epsilon ---
    etas_fixed = [0.001,0.005,0.01, 0.05, 0.1, 0.2, 0.5,1,2]
    eps_fixed = [0.001,0.005,0.01, 0.05, 0.1, 0.2, 0.5,0.7]
    fixed_results = np.zeros((len(etas_fixed), len(eps_fixed)))

    for i, eta in enumerate(etas_fixed):
        for j, eps in enumerate(eps_fixed):
            fixed_results[i, j] = run_match(
                agent_factory=lambda k: FixedRateQLearner(eta=eta, epsilon=eps, seed=None),
                goalkeeper_factory=lambda k: get_goal_keeper("biased", seed=None),
                num_matches=num_matches,
                kicks_per_match=50,
            )

    fixed_results_plot = fixed_results[::-1, :]

    plt.figure(figsize=(10, 8))
    sns.heatmap(fixed_results_plot,annot=False, xticklabels=eps_fixed,yticklabels=etas_fixed[::-1],cmap="YlGnBu")
    plt.title("FixedRateQLearner: Eta vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Eta")
    best_i, best_j = np.unravel_index(np.argmax(fixed_results), fixed_results.shape)
    best_i_plot = fixed_results.shape[0] - 1 - best_i

    plt.text(best_j+0.5 ,best_i_plot+0.5,"*",color="red",fontsize=50,ha="center",va="center")
    plt.show()

    #After hyperparameter tuning, the optimal parameter configurations were re-evaluated on independent runs in
    # order to obtain an unbiased estimate of the expected average reward:
    #MeanQLearner (ε=0.1)
    # PreferenceLearner (η=0.5)
    # FixedRateQLearner (η=0.05, ε=0.001)
    mean_meanq, std_meanq = unbiased_evaluation(
        agent_factory=lambda seed: MeanQLearner(epsilon=0.1, seed=seed),
        keeper_factory=lambda seed: get_goal_keeper("biased", seed=None),
        num_matches=10000,
        kicks_per_match=50,
    )
    print("MeanQLearner | epsilon=0.1 | mean:", mean_meanq)

    mean_pref, std_pref = unbiased_evaluation(
        agent_factory=lambda seed: PreferenceLearner(eta=0.5, seed=None),
        keeper_factory=lambda seed: get_goal_keeper("biased", seed=seed),
        num_matches=10000,
        kicks_per_match=50,
    )
    print("PreferenceLearner | eta=0.5 | mean:", mean_pref)

    mean_fixed, std_fixed = unbiased_evaluation(
        agent_factory=lambda seed: FixedRateQLearner(eta=0.05, epsilon=0.001, seed=seed),
        keeper_factory=lambda seed: get_goal_keeper("biased", seed=seed),
        num_matches=10000,
        kicks_per_match=50,)
    print("FixedRateQLearner | eta=0.05 epsilon=0.001 | mean:", mean_fixed)

    # Bonus simulation
    print("\nBonus simulation")

    mean_meanq, std_meanq = unbiased_evaluation(
        agent_factory=lambda seed: MeanQLearner(epsilon=0.1, seed=seed),
        keeper_factory=lambda seed: get_goal_keeper("adaptive", seed=None),
        num_matches=10000,
        kicks_per_match=50,
    )
    print("MeanQLearner | epsilon=0.1 | mean:", mean_meanq)

    mean_pref, std_pref = unbiased_evaluation(
        agent_factory=lambda seed: PreferenceLearner(eta=0.5, seed=None),
        keeper_factory=lambda seed: get_goal_keeper("adaptive", seed=seed),
        num_matches=10000,
        kicks_per_match=50,
    )
    print("PreferenceLearner | eta=0.5 | mean:", mean_pref)

    mean_fixed, std_fixed = unbiased_evaluation(
        agent_factory=lambda seed: FixedRateQLearner(eta=0.05, epsilon=0.001, seed=seed),
        keeper_factory=lambda seed: get_goal_keeper("adaptive", seed=seed),
        num_matches=10000,
        kicks_per_match=50, )
    print("FixedRateQLearner | eta=0.05 epsilon=0.001 | mean:", mean_fixed)




