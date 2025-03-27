import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from feature_net import FeatureNetwork
from successor_net import SuccessorNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
gamma = 0.99
lr = 2.5e-4
momentum = 0.95
epochs = 5000
feature_dim = 256
num_action_samples = 50  # Reduced for efficiency
batch_size = 128  # Increased for batch efficiency

# Epsilon-Greedy Parameters
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9997

# Load environment
env = gym.make("Reacher-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_steps = env.spec.max_episode_steps

# Initialize networks
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
w = nn.Parameter(torch.randn(feature_dim, device=device, requires_grad=True))  # Trainable reward weights

# Target network
successor_net_prev = SuccessorNetwork(feature_dim, action_dim).to(device)
successor_net_prev.load_state_dict(successor_net.state_dict())
successor_net_prev.eval()

# Optimizers (Only Using Momentum)
optimizer_f = optim.SGD(feature_net.parameters(), lr=lr, momentum=momentum)
optimizer_s = optim.SGD(successor_net.parameters(), lr=lr, momentum=momentum)
optimizer_w = optim.SGD([w], lr=lr, momentum=momentum)

# Replay Buffer
buffer_capacity = int(1e6)
replay_buffer = ReplayBuffer(buffer_capacity)

# Training Loop
for epoch in range(epochs):
    if epoch % 10 == 0:
        successor_net_prev.load_state_dict(successor_net.state_dict())

    state, info = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    terminated, truncated = False, False

    with tqdm(total=max_steps, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for step in range(max_steps):
            if terminated or truncated:
                break

            phi_s = feature_net(state)  # Compute feature representation
            
            with torch.no_grad():
                if np.random.rand() < epsilon:  
                    # Random action (exploration)
                    action = torch.tensor(
                        np.random.uniform(env.action_space.low, env.action_space.high, action_dim),
                        dtype=torch.float32, device=device
                    )
                else:  
                    # Greedy action (exploitation)
                    sampled_actions = torch.tensor(
                        np.random.uniform(env.action_space.low, env.action_space.high, (num_action_samples, action_dim)),
                        dtype=torch.float32, device=device
                    )
                    phi_s_expanded = phi_s.expand(num_action_samples, -1)
                    m_sDash_a = successor_net_prev(phi_s_expanded, sampled_actions)
                    q_values = (m_sDash_a @ w).squeeze(-1)
                    action = sampled_actions[q_values.argmax()]

            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)
            replay_buffer.push(state, action, reward, next_state, terminated)

            # Sample from replay buffer
            if len(replay_buffer) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                # Compute feature representations in batch
                phi_s_batch = feature_net(batch_states).squeeze(1)
                phi_next_s_batch = feature_net(batch_next_states).squeeze(1)
                # print(f"phi_s_batch shape: {phi_s_batch.shape}")  # Should be (batch_size, 256)
                # print(f"batch_actions shape: {batch_actions.shape}")  # Should be (batch_size, 2)
                # exit()
                # Compute successor representations in batch
                batch_actions = batch_actions.view(batch_size, action_dim)  # Ensure correct shape (batch_size, 2)
                m_sa_batch = successor_net(phi_s_batch, batch_actions)
                
                with torch.no_grad():
                    sampled_actions = torch.tensor(
                        np.random.uniform(env.action_space.low, env.action_space.high, (batch_size, num_action_samples, action_dim)),
                        dtype=torch.float32, device=device
                    )
                    phi_next_s_batch_expanded = phi_next_s_batch.unsqueeze(1).expand(-1, num_action_samples, -1)
                    m_sDash_a = successor_net_prev(phi_next_s_batch_expanded, sampled_actions)
                    q_values = (m_sDash_a @ w).squeeze(-1)
                    best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]
                
                # Compute target and loss
                target_M = phi_s_batch + gamma * best_m_sDash_a * (1 - batch_dones.unsqueeze(1))
                loss_sr = ((target_M - m_sa_batch) ** 2).mean()

                # Compute reward loss efficiently
                reward_pred_batch = (phi_s_batch @ w).squeeze()
                reward_loss = ((batch_rewards - reward_pred_batch) ** 2).mean()
                
                optimizer_f.zero_grad()
                optimizer_s.zero_grad()
                optimizer_w.zero_grad()
                
                loss_sr.backward(retain_graph=True)
                reward_loss.backward()
                
                optimizer_f.step()
                optimizer_s.step()
                optimizer_w.step()
            
            state = next_state
            total_reward += reward
            
            if step % 10 == 0:
                pbar.set_postfix(Total_Reward=total_reward)
            pbar.update(1)
    
    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f}")

# Save trained models
torch.save(feature_net.state_dict(), "feature_net.pth")
torch.save(successor_net.state_dict(), "successor_net.pth")
torch.save(w.detach().cpu(), "w.pth")
env.close()
