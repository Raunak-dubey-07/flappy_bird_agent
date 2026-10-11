# 🐦 Flappy Bird DQN Agent

A Deep Q-Network (DQN) reinforcement learning agent built with **PyTorch** that learns to play **Flappy Bird** using the [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium) environment.

The agent learns purely from trial and error: it observes the game state, picks flap / no-flap actions, and improves its policy using experience replay and a target network.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Hyperparameters](#-hyperparameters)
- [Outputs](#-outputs)
- [Known Limitations & Future Improvements](#-known-limitations--future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project implements a classic **DQN** pipeline from scratch:

- A small fully connected neural network approximates the Q-function.
- An **epsilon-greedy** policy balances exploration and exploitation.
- A **replay buffer** stores past transitions and breaks correlation between samples.
- A separate **target network** stabilizes the Bellman target during training.
- Hyperparameters are kept in a YAML file so experiments are easy to configure and reproduce.

The best-performing model during training is saved automatically and can be loaded later for rendered evaluation.

---

## ✨ Features

- Deep Q-Learning with experience replay and a target network
- Epsilon-greedy exploration with exponential decay
- Automatic GPU / Apple Silicon (MPS) / CPU device selection
- YAML-based hyperparameter configuration (multiple named parameter sets supported)
- Automatic checkpointing of the best model
- Training log of each new best reward
- Separate train and play (render) modes via a simple CLI

---

## 🧠 How It Works

1. **Observe** – The agent receives the state vector from the environment (12 features by default).
2. **Act** – With probability `epsilon` it picks a random action; otherwise it picks `argmax Q(s, a)`.
3. **Store** – Each transition `(state, action, next_state, reward, terminated)` goes into the replay memory.
4. **Learn** – A random mini-batch is sampled and the network is trained to minimize the MSE between the current Q-value and the Bellman target:

   ```
   target = r + γ · (1 − done) · max_a' Q_target(s', a')
   loss   = MSE(Q_policy(s, a), target)
   ```

5. **Sync** – The target network periodically copies the policy network's weights.
6. **Save** – Whenever an episode beats the previous best reward, the policy weights are saved.

### Network Architecture

| Layer | Details |
|-------|---------|
| Input | `state_dim` = 12 |
| Hidden | Linear(12 → 256) + ReLU |
| Output | Linear(256 → 2) (Q-values for *do nothing* / *flap*) |

---

## 📁 Project Structure

```
.
├── agent.py                # Training / evaluation loop, optimization step, CLI
├── dqn.py                  # DQN neural network definition
├── experience_replay.py    # Replay memory (deque-based buffer)
├── parameter.yaml          # Hyperparameter sets
├── requirements.txt        # Python dependencies
├── runs/                   # Generated: saved models (.pt) and logs (.log)
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- (Optional) a CUDA-capable GPU or Apple Silicon Mac for faster training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
torch
gymnasium
flappy-bird-gymnasium
pyyaml
```

---

## 🚀 Usage

### Train the agent

```bash
python agent.py flappybirdv0 --train
```

- `flappybirdv0` is the name of the parameter set in `parameter.yaml`.
- Training runs indefinitely; stop it with `Ctrl + C` once the reward is satisfactory.
- The best model is saved to `runs/flappybirdv0.pt`.

### Watch the trained agent play

```bash
python agent.py flappybirdv0
```

This loads `runs/flappybirdv0.pt` and runs the game with rendering enabled.

### Add your own experiment

Add a new block to `parameter.yaml`:

```yaml
my_experiment:
  env_id: FlappyBird-v0
  epsilon_init: 1
  epsilon_min: 0.05
  epsilon_decay: 0.999
  replay_memory_size: 50000
  mini_batch_size: 64
  network_sync_rate: 100
  alpha: 0.0005
  gamma: 0.99
  reward_threshold: 1000
```

Then run:

```bash
python agent.py my_experiment --train
```

---

## 🎛️ Hyperparameters

Defined in `parameter.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env_id` | `FlappyBird-v0` | Gymnasium environment ID |
| `epsilon_init` | `1` | Starting exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `0.9995` | Multiplicative decay applied after each episode |
| `replay_memory_size` | `100000` | Maximum number of stored transitions |
| `mini_batch_size` | `32` | Batch size for each optimization step |
| `network_sync_rate` | `10` | Environment steps between target network updates |
| `alpha` | `0.001` | Learning rate (Adam optimizer) |
| `gamma` | `0.99` | Discount factor |
| `reward_threshold` | `1000` | Episode is ended once this reward is reached |

---

## 📤 Outputs

Created automatically inside the `runs/` folder:

| File | Description |
|------|-------------|
| `runs/<param_set>.pt` | Weights of the best-performing policy network |
| `runs/<param_set>.log` | Log of every new best reward and the episode it occurred in |

Console output during training looks like:

```
Episode 42 finished with reward 12.9 and epsilon 0.81
New best reward 12.9. Model saved to runs/flappybirdv0.pt
```

---


---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) for the environment
- [Gymnasium](https://gymnasium.farama.org/) for the RL environment API
- [PyTorch](https://pytorch.org/) for the deep learning framework
- *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013) for the original DQN idea
