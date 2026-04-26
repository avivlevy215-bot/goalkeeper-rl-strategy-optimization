from abc import abstractmethod
import numpy as np

class GoalKeeper:
    """
    A goal keeper environment class.
    
    In each turn, the goal keeper predecides
    whether to dive left or right.

    If the kicker shoots in the same direction as the goal keeper,
    the goal keeper saves the goal and the kicker
    receives a reward of -1.

    Otherwise, the kicker scores a goal and receives a reward of +1.
    """

    def __init__(self):
        self.kick_history = []
        self.dive_history = []
        self.total_reward = 0
        
    def step(self, kicker_action):
        """
        Take a step in the environment.

        Parameters:
        kicker_action (str): The action taken by the kicker ('left' or 'right').

        returns:
        goal_keeper_decision (str): The action taken by the goal keeper ('left' or 'right').
        reward (int): The reward received by the kicker (+1 for scoring, -1 for being saved).
        """
        
        assert kicker_action in ['left', 'right'], "kicker_action must be 'left' or 'right'"
        goal_keeper_decision = self.predecide_goal_keeper_action()

        if kicker_action == goal_keeper_decision:
            reward = -1  # Goal keeper saves the goal
        else:
            reward = 1   # Kicker scores a goal
        
        self.total_reward += reward
        self.kick_history.append(kicker_action)
        self.dive_history.append(goal_keeper_decision)
        return goal_keeper_decision, reward
    
    @abstractmethod
    def predecide_goal_keeper_action(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class Biased_Goal_Keeper(GoalKeeper):
    """
    A biased goal keeper environment class.

    The goal keepers has a probability theta of diving left.
    """
    def __init__(self, theta=0.5, seed=None):
        super().__init__()
        self.theta = theta
        self.rng = np.random.default_rng(seed)
    
    def predecide_goal_keeper_action(self):
        return 'left' if self.rng.random() < self.theta else 'right'

#Bonos_class
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
        self.beta  = self.beta0  + self.decay * (self.beta  - self.beta0)

        if kicker_action == "left":
            self.alpha += 1.0
        else:
            self.beta += 1.0
        return decision, reward


def get_goal_keeper(goal_keeper_type, seed=None):
    """"
    Factory method to get a goal keeper instance based on the index.
    
    args:
    goal_keeper_type (str): The type of goal keeper ('biased', 'unbiased', 'adversarial').

    """

    rng = np.random.default_rng(seed)

    assert isinstance(goal_keeper_type, str), "goal_keeper_type must be a string"
    
    if goal_keeper_type == 'biased':
        return Biased_Goal_Keeper(theta=rng.random()*0.3 + 0.35, seed=seed)  # biased between [0.35, 0.65]
    elif goal_keeper_type == "adaptive":
        return Adaptive_Goal_Keeper(seed=seed)
    else:
        raise ValueError(f"Unknown goal keeper type: {goal_keeper_type}")
    

