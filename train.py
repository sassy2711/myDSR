import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from feature_net import FeatureNetwork
from successor_net import SuccessorNetwork
from replay_buffer import ReplayBuffer
from intrinsic_reward_predictor import IntrinsicRewardPredictor

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
gamma = 0.99
lr = 2.5e-4
momentum = 0.95
epochs = 10000
feature_dim = 256
num_action_samples = 50  # Reduced for efficiency
batch_size = 128  # Increased for batch efficiency

# Epsilon-Greedy Parameters
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999744

# Load environment
env = gym.make("Reacher-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_steps = env.spec.max_episode_steps

feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
intrinsic_reward_net = IntrinsicRewardPredictor(feature_dim, state_dim).to(device)

# Recommended: Kaiming Uniform Init for trainable reward weights
w = nn.Parameter(torch.empty(feature_dim, device=device))
nn.init.kaiming_uniform_(w.unsqueeze(0), nonlinearity='relu')
w.requires_grad_()

# Target network
successor_net_prev = SuccessorNetwork(feature_dim, action_dim).to(device)
successor_net_prev.load_state_dict(successor_net.state_dict())
successor_net_prev.eval()

# Optimizers (Only Using Momentum)
optimizer_theta = optim.SGD(feature_net.parameters(), lr=lr, momentum=momentum)
optimizer_alpha = optim.SGD(successor_net.parameters(), lr=lr, momentum=momentum)
optimizer_theta_tilde = optim.SGD(intrinsic_reward_net.parameters(), lr=lr, momentum=momentum)
optimizer_w = optim.SGD([w], lr=lr, momentum=momentum)

# Replay Buffer
buffer_capacity = int(1e6)
replay_buffer = ReplayBuffer(buffer_capacity)

# Training Loop
for epoch in range(epochs):
    epoch_l_r = []
    epoch_l_a = []
    epoch_loss_sr = []

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
            
            action = None
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
                #m_sa_batch = successor_net(phi_s_batch, batch_actions)
                
                with torch.no_grad():
                    sampled_actions = torch.tensor(
                        np.random.uniform(env.action_space.low, env.action_space.high, (batch_size, num_action_samples, action_dim)),
                        dtype=torch.float32, device=device
                    )
                    phi_next_s_batch_expanded = phi_next_s_batch.unsqueeze(1).expand(-1, num_action_samples, -1)
                    m_sDash_a = successor_net_prev(phi_next_s_batch_expanded, sampled_actions)
                    q_values = (m_sDash_a @ w).squeeze(-1)
                    best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]

                # Compute reward loss efficiently
                reward_pred_batch = (phi_s_batch @ w).squeeze()
                l_r = ((batch_rewards.squeeze(-1) - reward_pred_batch) ** 2).mean()

                # phi_s_batch: (batch_size, feature_dim)
                # batch_states: (batch_size, state_dim)
                reconstructed_states = intrinsic_reward_net(phi_s_batch)

                # Loss: mean squared error between reconstructed and actual state
                l_a = ((reconstructed_states - batch_states.squeeze(1)) ** 2).mean()
                
                # After computing l_r and l_a
                epoch_l_r.append(l_r.item())
                epoch_l_a.append(l_a.item())

                reward_loss = l_r + l_a
                # Prevent NaNs/Infs
                if torch.isnan(l_r) or torch.isinf(l_r):
                    print("❌ Skipping l_r due to instability")
                    continue
                if torch.isnan(l_a) or torch.isinf(l_a):
                    print("❌ Skipping l_a due to instability")
                    continue
                if torch.isnan(reward_loss) or torch.isinf(reward_loss):
                    print("❌ Skipping reward_loss due to instability")
                    continue
                # === Step 1: Optimize reward prediction loss (only w and feature_net) ===
                # optimizer_theta.zero_grad()
                # optimizer_w.zero_grad()

                # l_r.backward()
                # optimizer_theta.step()
                # optimizer_w.step()
                optimizer_theta.zero_grad()
                optimizer_w.zero_grad()
                optimizer_theta_tilde.zero_grad()
                reward_loss.backward()

                # 🔒 Gradient Clipping for feature_net and w
                torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=50)
                torch.nn.utils.clip_grad_norm_(intrinsic_reward_net.parameters(), max_norm=50)
                torch.nn.utils.clip_grad_norm_([w], max_norm=50)

                optimizer_theta.step()
                optimizer_w.step()
                optimizer_theta_tilde.step()

                # # === Step 2: Optimize successor representation loss (only successor_net) ===
                # optimizer_alpha.zero_grad()

                # # Detach w and phi_s_batch to prevent backprop into them
                # phi_s_batch_detached = phi_s_batch.detach()
                # best_m_sDash_a_detached = best_m_sDash_a.detach()
                # target_M = phi_s_batch_detached + gamma * best_m_sDash_a_detached * (1 - batch_dones.unsqueeze(1))

                # m_sa_batch = successor_net(phi_s_batch_detached, batch_actions)
                # loss_sr = ((target_M - m_sa_batch) ** 2).mean()

                # loss_sr.backward()
                # optimizer_alpha.step()
                optimizer_alpha.zero_grad()

                # Detach w and phi_s_batch to prevent backprop into them
                phi_s_batch_detached = phi_s_batch.detach()
                best_m_sDash_a_detached = best_m_sDash_a.detach()
                target_M = phi_s_batch_detached + gamma * best_m_sDash_a_detached * (1 - batch_dones)

                m_sa_batch = successor_net(phi_s_batch_detached, batch_actions)
                loss_sr = ((target_M - m_sa_batch) ** 2).mean()

                # After computing loss_sr
                epoch_loss_sr.append(loss_sr.item())

                if torch.isnan(loss_sr) or torch.isinf(loss_sr):
                    print("❌ Skipping loss_sr due to instability")
                    continue

                loss_sr.backward()

                # 🔒 Gradient Clipping for successor_net
                torch.nn.utils.clip_grad_norm_(successor_net.parameters(), max_norm=50)

                optimizer_alpha.step()

            
            state = next_state
            total_reward += reward


            if step % 10 == 0:
                pbar.set_postfix(Total_Reward=total_reward)
            pbar.update(1)
    
    avg_l_r = np.mean(epoch_l_r) if epoch_l_r else float('nan')
    avg_l_a = np.mean(epoch_l_a) if epoch_l_a else float('nan')
    avg_loss_sr = np.mean(epoch_loss_sr) if epoch_loss_sr else float('nan')
        
    print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f} | "
        f"Loss_r: {avg_l_r:.4f} | Loss_a: {avg_l_a:.4f} | Loss_SR: {avg_loss_sr:.4f}")

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f}")

# Save trained models
torch.save(feature_net.state_dict(), "feature_net.pth")
torch.save(successor_net.state_dict(), "successor_net.pth")
torch.save(intrinsic_reward_net.state_dict(), "intrinsic_reward_net.pth")
torch.save(w.detach().cpu(), "w.pth")
env.close()

