# Goalkeeper RL Strategy Optimization

Reinforcement Learning simulation for optimizing penalty kicks.
# Goalkeeper RL Strategy Optimization

This project simulates a penalty-kick scenario using Reinforcement Learning (RL) techniques to optimize decision-making against different types of goalkeepers.

## 🚀 Overview

The environment models a simple game where a kicker chooses to shoot left or right, while a goalkeeper attempts to predict and block the shot. The goal is to evaluate how different learning algorithms adapt and perform under uncertainty.

## 🧠 Implemented Algorithms

* **Mean Q-Learning** (sample-average method)
* **Fixed Rate Q-Learning**
* **Preference-Based Learning (Policy Gradient style)**

## 🥅 Goalkeeper Types

* **Biased Goalkeeper** – chooses a direction based on a fixed probability
* **Adaptive Goalkeeper** – learns and adapts to the kicker’s behavior using Thompson Sampling

## 📊 Features

* Hyperparameter tuning (epsilon, eta)
* Performance evaluation across multiple simulations
* Visualization using Matplotlib and Seaborn
* Comparison between different learning strategies

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

```bash
python main.py
```

Optional test run:

```bash
python test_run.py
```

## 📁 Project Structure

```
goalkeeper-rl/
│── environment.py
│── learners.py
│── main.py
│── test_run.py
│── requirements.txt
│── README.md
```

## 📌 Key Insights

* Different exploration strategies significantly impact performance
* Adaptive environments require more robust learning approaches
* Hyperparameter tuning is critical for optimal results

## 📈 Future Improvements

* Add Deep Reinforcement Learning (DQN / Policy Networks)
* Extend action space beyond binary decisions
* Introduce real-world data or more complex simulations

## 👨‍💻 Author

Aviv Levy
