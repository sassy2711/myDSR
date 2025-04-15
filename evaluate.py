import gymnasium as gym
import torch
import numpy as np
from feature_net import FeatureNetwork
from successor_net import SuccessorNetwork

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load environment
env = gym.make("Reacher-v5")  # Enable rendering
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Feature & Successor network parameters
feature_dim = 256

# Load trained models
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)

feature_net.load_state_dict(torch.load("feature_net.pth", map_location=device, weights_only=True))
successor_net.load_state_dict(torch.load("successor_net.pth", map_location=device, weights_only=True))


feature_net.eval()
successor_net.eval()

gamma = 0.99
num_episodes = 5  # Number of episodes for evaluation
num_action_samples = 10

# Load learned reward weight vector
w = torch.load("w.pth", map_location=device, weights_only=True)
#print("w stats:", w.min(), w.max(), torch.isnan(w).any(), torch.isinf(w).any())

for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    terminated, truncated = False, False

    while not (terminated or truncated):
        phi_s = feature_net(state)

        # Sample multiple actions to find the best one
        sampled_actions = torch.tensor(
            np.random.uniform(env.action_space.low, env.action_space.high, (num_action_samples, action_dim)),
            dtype=torch.float32, device=device
        )

        best_action = None
        max_q_value = float('-inf')

        for i in sampled_actions:
            a = i.unsqueeze(0)
            m_s_a = successor_net(phi_s, a)
            # print("m_s_a stats:", torch.isnan(m_s_a).any(), torch.isinf(m_s_a).any())
            q_value = m_s_a @ w  # ✅ Use the learned weight vector
            print(q_value.shape)
            print(m_s_a.shape)
            if q_value > max_q_value:
                #print(1)
                max_q_value = q_value
                best_action = a

        action = best_action.detach().cpu().numpy().flatten()

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)

        total_reward += reward
        state = next_state  # Move to next state

    print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f}")

env.close()
