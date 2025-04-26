import gymnasium as gym
import torch
import numpy as np
import os
from reward_net import RewardNetwork
from successor_net import SuccessorNetwork
from feature_net import FeatureNetwork

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories for saving videos
video_folder = './videos_eval'
os.makedirs(video_folder, exist_ok=True)
record_interval = 10  # Record every 50 episodes

# Load environment to get state/action space
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.close()

# Network parameters
feature_dim = 128

# Load trained models
#reward_net = RewardNetwork(state_dim).to(device)
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
#reward_net.load_state_dict(torch.load("reward_net.pth", map_location=device))
successor_net.load_state_dict(torch.load("successor_net.pth", map_location=device))
feature_net.load_state_dict(torch.load("feature_net.pth", map_location=device))
feature_net.eval()
successor_net.eval()
# Load learned reward weight vector
w = torch.load("w.pth", map_location=device)  # shape: [feature_dim]
w = w.to(device)  # just to ensure it's on the right device

# # Load learned reward weight vector
# w = torch.load("w.pth", map_location=device)  # shape: [feature_dim]

# Evaluation settings
num_episodes = 100

for episode in range(num_episodes):
    if episode % record_interval == 0:
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda x: x == 0,
            name_prefix=f"eval_ep_{episode}"
        )
    else:
        env = gym.make("MountainCar-v0")

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    terminated, truncated = False, False

    while not (terminated or truncated):
        # Get reward features
        with torch.no_grad():
            #_, phi_s, w = reward_net(state)
            phi_s = feature_net(state)

            # Evaluate Q-values for all discrete actions
            action_candidates = torch.eye(action_dim, device=device)
            phi_s_exp = phi_s.expand(action_dim, -1)
            m_s_a = successor_net(phi_s_exp, action_candidates)
            q_values = (m_s_a @ w).squeeze(-1)
            best_action = q_values.argmax().item()

        # Take the best action
        next_state, reward, terminated, truncated, _ = env.step(best_action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward += reward
        state = next_state

    print(f"🎬 Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f}")
    env.close()
