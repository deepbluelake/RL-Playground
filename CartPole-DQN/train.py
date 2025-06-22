import gym
import torch
import numpy as np
from dqn_agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(state_size, action_size, device)

n_episodes = 500
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

scores = []
scores_window = deque(maxlen=100)
epsilon = eps_start

for episode in range(1, n_episodes + 1):
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    total_reward = 0
    for t in range(max_t):
        action = agent.act(state, epsilon)
        step_out = env.step(action)
        next_state, reward, terminated, truncated, _ = step_out
        done = terminated or truncated

        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    scores_window.append(total_reward)
    scores.append(total_reward)
    epsilon = max(eps_end, eps_decay * epsilon)

    print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}")

    if np.mean(scores_window) >= 195.0:
        print(f"\nEnvironment solved in {episode} episodes!")
        torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
        break

torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
env.close()

# Plot score curve
import matplotlib.pyplot as plt

# Plotting training results
plt.figure(figsize=(10, 5))
plt.plot(scores, label='Score per Episode')
plt.plot([np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))], label='Moving Average (100)', color='orange')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Progress')
plt.legend()
plt.grid(True)
plt.savefig("training_plot.png")  # Save the plot as an image
plt.show()  # Show the plot (optional)

