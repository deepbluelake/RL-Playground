import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network for approximating Q-values
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.model(x)

# DQN Agent with Double DQN and delayed target updates
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Q-networks
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-3)

        # Replay buffer
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Training control
        self.gamma = 0.99
        self.update_every = 4
        self.step_count = 0
        self.target_update_freq = 1000  # hard update every 1000 steps

    def step(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))
        self.step_count += 1

        if len(self.memory) > self.batch_size and self.step_count % self.update_every == 0:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

        # Delayed target network update
        if self.step_count % self.target_update_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.1):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.qnetwork_local(state)
            return q_values.argmax().item()
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = self.extract_tensors(experiences)

        # DOUBLE DQN
        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def extract_tensors(self, experiences):
        batch = self.experience(*zip(*experiences))
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones
