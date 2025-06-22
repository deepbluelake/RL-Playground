# RL Playground 🎮

This repository contains mini-projects focused on reinforcement learning algorithms implemented using Python and PyTorch.

## 📁 Projects

### 🔹 [CartPole-DQN](./CartPole-DQN)

Solves the CartPole-v1 environment using Deep Q-Network (DQN) with experience replay, target networks, and epsilon-greedy strategy.

- ✅ Trained model reaches an average score of ~197.
- 📈 Training curve and demo video included.

---

## 📦 Setup

```bash
conda create -n dqn_cartpole python=3.10
conda activate dqn_cartpole
pip install -r CartPole-DQN/requirements.txt
