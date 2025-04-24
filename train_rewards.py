# train_rewards.py

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from reward_net import RewardNetwork

# Setup for CartPole
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # Should be 4
reward_net = RewardNetwork(state_dim)
optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)  # slightly higher LR is okay here
loss_fn = nn.MSELoss()
epochs = 1000  # You can keep it at 10000 if you want to watch convergence long-term

for episode in range(epochs):
    state, _ = env.reset()
    done = False
    episode_states = []
    episode_rewards = []

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode_states.append(state)
        episode_rewards.append(reward)  # will always be 1.0 unless early stop

        state = next_state
        done = terminated or truncated

    # Convert to tensors
    states = torch.tensor(np.array(episode_states), dtype=torch.float32)
    rewards = torch.tensor(np.array(episode_rewards), dtype=torch.float32).unsqueeze(1)

    pred_rewards, _, _ = reward_net(states)
    loss = loss_fn(pred_rewards, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode} | Steps: {len(episode_states)} | Loss: {loss.item():.4f}")

env.close()
