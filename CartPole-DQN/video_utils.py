import gym
import torch
import numpy as np
import os
import cv2

from dqn_agent import DQNAgent

def record_video(env_name, checkpoint_path, video_path="cartpole_output.mp4", episodes=1):
    env = gym.make(env_name, render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_size, action_size, device)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=device))

    frames = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            frames.append(frame)
            action = agent.act(state, eps=0.0)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

    env.close()

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Video saved to {video_path}")
