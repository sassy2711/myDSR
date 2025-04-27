import gymnasium as gym
import torch
import pickle
import os
from successor_net import SuccessorNetwork
from feature_net import FeatureNetwork

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment setup
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.close()

# Feature dimension (same as training)
feature_dim = 128

# Load models
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
feature_net.load_state_dict(torch.load("feature_net.pth", map_location=device))
successor_net.load_state_dict(torch.load("successor_net.pth", map_location=device))
feature_net.eval()
successor_net.eval()

# Load learned reward weight vector
w = torch.load("w.pth", map_location=device).to(device)

# Trajectory generation
num_episodes = 500
trajectories = []

for episode in range(num_episodes):
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_trajectory = []
    timestep = 0
    terminated, truncated = False, False

    while not (terminated or truncated):
        with torch.no_grad():
            phi_s = feature_net(state)
            action_candidates = torch.eye(action_dim, device=device)
            phi_s_exp = phi_s.expand(action_dim, -1)
            m_s_a = successor_net(phi_s_exp, action_candidates)
            q_values = (m_s_a @ w).squeeze(-1)
            best_action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(best_action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        # Save (reward, state, action, timestep)
        episode_trajectory.append([
            float(reward),
            state.squeeze(0).cpu().tolist(),
            int(best_action),
            int(timestep)
        ])

        state = next_state_tensor
        timestep += 1

    trajectories.append(episode_trajectory)
    env.close()

    # 🔥 Print progress every episode
    print(f"Episode {episode + 1}/{num_episodes} completed.")

print(f"✅ Finished generating {len(trajectories)} trajectories.")

# Save the trajectories
save_path = "cartpole_trajectories.pkl"
with open(save_path, "wb") as f:
    pickle.dump(trajectories, f)

print(f"💾 Saved trajectories to {save_path}")
