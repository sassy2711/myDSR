# import gymnasium as gym
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from feature_net import FeatureNetwork
# from successor_net import SuccessorNetwork
# from replay_buffer import ReplayBuffer
# from intrinsic_reward_predictor import IntrinsicRewardPredictor

# def weights_init_kaiming(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Hyperparameters
# gamma = 0.99
# lr = 2.5e-4
# momentum = 0.95
# epochs = 10000
# feature_dim = 256
# num_action_samples = 4  # All possible discrete actions
# batch_size = 128

# # Epsilon-Greedy Parameters
# epsilon = 1.0
# epsilon_min = 0.1
# epsilon_decay = 0.999744

# # Load environment
# env = gym.make("LunarLander-v3")  # Discrete action space
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# max_steps = 1000

# feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
# successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
# intrinsic_reward_net = IntrinsicRewardPredictor(feature_dim, state_dim).to(device)

# # Reward weight vector
# w = nn.Parameter(torch.empty(feature_dim, device=device))
# nn.init.kaiming_uniform_(w.unsqueeze(0), nonlinearity='relu')
# w.requires_grad_()

# # Target network
# successor_net_prev = SuccessorNetwork(feature_dim, action_dim).to(device)
# successor_net_prev.load_state_dict(successor_net.state_dict())
# successor_net_prev.eval()

# # Optimizers
# optimizer_theta = optim.SGD(feature_net.parameters(), lr=lr, momentum=momentum)
# optimizer_alpha = optim.SGD(successor_net.parameters(), lr=lr, momentum=momentum)
# optimizer_theta_tilde = optim.SGD(intrinsic_reward_net.parameters(), lr=lr, momentum=momentum)
# optimizer_w = optim.SGD([w], lr=lr, momentum=momentum)

# # Replay Buffer
# buffer_capacity = int(1e6)
# replay_buffer = ReplayBuffer(buffer_capacity)

# # One-hot encode function for discrete actions
# def one_hot(actions, num_classes):
#     return torch.eye(num_classes, device=actions.device)[actions]

# # Training Loop
# for epoch in range(epochs):
#     epoch_l_r = []
#     epoch_l_a = []
#     epoch_loss_sr = []

#     if epoch % 10 == 0:
#         successor_net_prev.load_state_dict(successor_net.state_dict())
#     state, info = env.reset()
#     state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

#     total_reward = 0
#     terminated, truncated = False, False

#     with tqdm(total=max_steps, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
#         for step in range(max_steps):
#             if terminated or truncated:
#                 break

#             phi_s = feature_net(state)

#             # Action Selection
#             with torch.no_grad():
#                 if np.random.rand() < epsilon:
#                     action = np.random.randint(0, action_dim)
#                 else:
#                     # Compute Q-values for all discrete actions
#                     action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
#                     phi_s_expanded = phi_s.expand(action_dim, -1)
#                     m_s_a = successor_net_prev(phi_s_expanded, action_candidates)

#                     q_values = (m_s_a @ w).squeeze(-1)

#                     action = q_values.argmax().item()

#             next_state, reward, terminated, truncated, info = env.step(action)
#             next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)

#             replay_buffer.push(state, torch.tensor([action], device=device), reward, next_state, terminated)

#             # Sample training batch
#             if len(replay_buffer) >= batch_size:
#                 batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
#                 batch_states = batch_states.squeeze(1)
#                 phi_s_batch = feature_net(batch_states).squeeze(1)
#                 phi_next_s_batch = feature_net(batch_next_states).squeeze(1)
                    
#                 batch_actions_oh = one_hot(batch_actions.squeeze(-1).long(), action_dim)

#                 # Compute target successor representations
#                 with torch.no_grad():
#                     action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
#                     phi_next_s_batch_expanded = phi_next_s_batch.unsqueeze(1).expand(-1, action_dim, -1)
#                     action_candidates_expanded = action_candidates.unsqueeze(0).expand(batch_size, -1, -1)

#                     m_sDash_a = successor_net_prev(phi_next_s_batch_expanded, action_candidates_expanded)

#                     q_values = (m_sDash_a @ w).squeeze(-1)

#                     best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]

#                 # Reward prediction loss
#                 reward_pred_batch = (phi_s_batch @ w).unsqueeze(-1)

#                 l_r = ((batch_rewards.squeeze(-1) - reward_pred_batch) ** 2).mean()

#                 # Autoencoder loss
#                 reconstructed_states = intrinsic_reward_net(phi_s_batch)
#                 l_a = ((reconstructed_states - batch_states.squeeze(1)) ** 2).mean()

#                 epoch_l_r.append(l_r.item())
#                 epoch_l_a.append(l_a.item())

#                 reward_loss = l_r + l_a

#                 if torch.isnan(reward_loss) or torch.isinf(reward_loss):
#                     print("❌ Skipping reward_loss due to instability")
#                     continue

#                 optimizer_theta.zero_grad()
#                 optimizer_w.zero_grad()
#                 optimizer_theta_tilde.zero_grad()
#                 reward_loss.backward()

#                 torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=50)
#                 torch.nn.utils.clip_grad_norm_(intrinsic_reward_net.parameters(), max_norm=50)
#                 torch.nn.utils.clip_grad_norm_([w], max_norm=50)

#                 optimizer_theta.step()
#                 optimizer_w.step()
#                 optimizer_theta_tilde.step()

#                 # Successor representation loss
#                 optimizer_alpha.zero_grad()

#                 phi_s_batch_detached = phi_s_batch.detach()
#                 best_m_sDash_a_detached = best_m_sDash_a.detach()
#                 target_M = phi_s_batch_detached + gamma * best_m_sDash_a_detached * (1 - batch_dones)

#                 m_sa_batch = successor_net(phi_s_batch_detached, batch_actions_oh)

#                 loss_sr = ((target_M - m_sa_batch) ** 2).mean()

#                 epoch_loss_sr.append(loss_sr.item())

#                 if torch.isnan(loss_sr) or torch.isinf(loss_sr):
#                     print("❌ Skipping loss_sr due to instability")
#                     continue

#                 loss_sr.backward()
#                 torch.nn.utils.clip_grad_norm_(successor_net.parameters(), max_norm=50)
#                 optimizer_alpha.step()

#             state = next_state
#             total_reward += reward

#             if step % 10 == 0:
#                 pbar.set_postfix(Total_Reward=total_reward)
#             pbar.update(1)

#     if(epoch == 1):
#         print(f"Shape of batch_rewards: {batch_rewards.shape}")
#         print(f"Shape of reward_pred_batch: {reward_pred_batch.shape}")
#         print(f"Shape of reconstructed_states: {reconstructed_states.shape}")
#         print(f"Shape of batch_states: {batch_states.shape}")
#         print(f"Shape of target_M: {target_M.shape}")
#         print(f"Shape of m_sa_batch: {m_sa_batch.shape}")
#         print(f"Shape of w: {w.shape}")
#         print(f"Shape of phi_s_batch: {phi_s_batch.shape}")


#     avg_l_r = np.mean(epoch_l_r) if epoch_l_r else float('nan')
#     avg_l_a = np.mean(epoch_l_a) if epoch_l_a else float('nan')
#     avg_loss_sr = np.mean(epoch_loss_sr) if epoch_loss_sr else float('nan')

#     print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f} | "
#           f"Loss_r: {avg_l_r:.4f} | Loss_a: {avg_l_a:.4f} | Loss_SR: {avg_loss_sr:.4f}")

#     epsilon = max(epsilon_min, epsilon * epsilon_decay)

# # Save models
# torch.save(feature_net.state_dict(), "feature_net.pth")
# torch.save(successor_net.state_dict(), "successor_net.pth")
# torch.save(intrinsic_reward_net.state_dict(), "intrinsic_reward_net.pth")
# torch.save(w.detach().cpu(), "w.pth")
# env.close()

import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from feature_net import FeatureNetwork  # Ensure feature_net uses smaller architecture
from successor_net import SuccessorNetwork
from replay_buffer import ReplayBuffer
from intrinsic_reward_predictor import IntrinsicRewardPredictor
import os

video_folder = './videos'
os.makedirs(video_folder, exist_ok=True)
record_interval = 50  # Record every 50 epochs


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
gamma = 0.985
epochs = 1500
feature_dim = 128
num_action_samples = 4  # All possible discrete actions
batch_size = 64

# Epsilon-Greedy Parameters
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.997

# Learning Rates
lr_theta = 1e-4
lr_alpha = 2e-4
lr_tilde = 1e-4
lr_w = 5e-5

# Load environment
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_steps = 800

# Networks
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
intrinsic_reward_net = IntrinsicRewardPredictor(feature_dim, state_dim).to(device)

# Reward weight vector
w = nn.Parameter(torch.empty(feature_dim, device=device))
nn.init.kaiming_uniform_(w.unsqueeze(0), nonlinearity='relu')
w.requires_grad_()

# Target network
successor_net_prev = SuccessorNetwork(feature_dim, action_dim).to(device)
successor_net_prev.load_state_dict(successor_net.state_dict())
successor_net_prev.eval()

# Optimizers
optimizer_theta = optim.SGD(feature_net.parameters(), lr=lr_theta, momentum=0.95)
optimizer_alpha = optim.SGD(successor_net.parameters(), lr=lr_alpha, momentum=0.95)
optimizer_theta_tilde = optim.SGD(intrinsic_reward_net.parameters(), lr=lr_tilde, momentum=0.95)
optimizer_w = optim.SGD([w], lr=lr_w, momentum=0.95)

# Replay Buffer
buffer_capacity = int(3e4)
replay_buffer = ReplayBuffer(buffer_capacity)

# One-hot encoding
def one_hot(actions, num_classes):
    return torch.eye(num_classes, device=actions.device)[actions]

def soft_update(target_net, source_net, tau=0.005):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
#print(w.shape)
# Training loop
for epoch in range(epochs):
    if hasattr(env, 'close'):
        env.close()

    if epoch % record_interval == 0:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=f"epoch_{epoch}"
        )
    else:
        env = gym.make("LunarLander-v3")

    epoch_l_r = []
    epoch_l_a = []
    epoch_loss_sr = []

    soft_update(successor_net_prev, successor_net, tau=0.005)

    state, _ = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    terminated, truncated = False, False

    with tqdm(total=max_steps, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for step in range(max_steps):
            if terminated or truncated:
                break

            phi_s = feature_net(state)

            # Epsilon-greedy action selection
            with torch.no_grad():
                #print("Epsilon: ", epsilon)
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                    phi_s_exp = phi_s.expand(action_dim, -1)
                    m_s_a = successor_net_prev(phi_s_exp, action_candidates)
                    q_values = (m_s_a @ w).squeeze(-1)
                    action = q_values.argmax().item()
                    # print(m_s_a.shape)
                    # print(q_values.shape)
                    # print(action_candidates.shape)
                    # print(phi_s_exp.shape)
                    # print(phi_s.shape)
                    #print(action.shape)

            next_state, reward, terminated, truncated, _ = env.step(action)
            reward /= 100.0  # Reward scaling

            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)

            replay_buffer.push(state, torch.tensor([action], device=device), reward, next_state, terminated)

            # Training step
            if len(replay_buffer) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                batch_states = batch_states.squeeze(1)
                phi_s_batch = feature_net(batch_states).squeeze(1)
                phi_next_s_batch = feature_net(batch_next_states).squeeze(1)
                batch_actions_oh = one_hot(batch_actions.squeeze(-1).long(), action_dim)

                with torch.no_grad():
                    action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                    phi_next_exp = phi_next_s_batch.unsqueeze(1).expand(-1, action_dim, -1)  # [B, A, F]
                    action_exp = action_candidates.unsqueeze(0).expand(batch_size, -1, -1)   # [B, A, A_dim]

                    # 🔄 Flatten inputs
                    B, A, F = phi_next_exp.shape
                    _, _, A_dim = action_exp.shape
                    phi_next_exp_flat = phi_next_exp.reshape(B * A, F)       # [B*A, F]
                    action_exp_flat = action_exp.reshape(B * A, A_dim)       # [B*A, A_dim]

                    # Run network and reshape back
                    m_sDash_a_flat = successor_net_prev(phi_next_exp_flat, action_exp_flat)  # [B*A, F]
                    m_sDash_a = m_sDash_a_flat.view(B, A, F)                                 # 🔙 [B, A, F]

                    q_values = (m_sDash_a @ w).squeeze(-1)                                   # [B, A]
                    best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]  # [B, F]

                reward_pred_batch = (phi_s_batch @ w).unsqueeze(-1)
                l_r = ((batch_rewards.squeeze(-1) - reward_pred_batch) ** 2).mean()

                reconstructed_states = intrinsic_reward_net(phi_s_batch)
                l_a = ((reconstructed_states - batch_states.squeeze(1)) ** 2).mean()

                epoch_l_r.append(l_r.item())
                epoch_l_a.append(l_a.item())

                reward_loss = l_r + l_a
                if torch.isnan(reward_loss) or torch.isinf(reward_loss):
                    print("❌ Skipping reward_loss due to instability")
                    continue

                optimizer_theta.zero_grad()
                optimizer_w.zero_grad()
                optimizer_theta_tilde.zero_grad()
                reward_loss.backward()
                torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=50)
                torch.nn.utils.clip_grad_norm_(intrinsic_reward_net.parameters(), max_norm=50)
                torch.nn.utils.clip_grad_norm_([w], max_norm=50)
                optimizer_theta.step()
                optimizer_w.step()
                optimizer_theta_tilde.step()

                optimizer_alpha.zero_grad()
                phi_s_batch_detached = phi_s_batch.detach()
                best_m_sDash_a_detached = best_m_sDash_a.detach()
                target_M = phi_s_batch_detached + gamma * best_m_sDash_a_detached * (1 - batch_dones)

                m_sa_batch = successor_net(phi_s_batch_detached, batch_actions_oh)
                loss_sr = ((target_M - m_sa_batch) ** 2).mean()
                epoch_loss_sr.append(loss_sr.item())

                if torch.isnan(loss_sr) or torch.isinf(loss_sr):
                    print("❌ Skipping loss_sr due to instability")
                    continue

                loss_sr.backward()
                torch.nn.utils.clip_grad_norm_(successor_net.parameters(), max_norm=50)
                optimizer_alpha.step()


            state = next_state
            total_reward += reward * 100.0  # Scale reward back for logging

            if step % 10 == 0:
                pbar.set_postfix(Total_Reward=total_reward)
            pbar.update(1)

    avg_l_r = np.mean(epoch_l_r) if epoch_l_r else float('nan')
    avg_l_a = np.mean(epoch_l_a) if epoch_l_a else float('nan')
    avg_loss_sr = np.mean(epoch_loss_sr) if epoch_loss_sr else float('nan')

    print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f} | "
          f"Loss_r: {avg_l_r:.4f} | Loss_a: {avg_l_a:.4f} | Loss_SR: {avg_loss_sr:.4f}")

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Save models
torch.save(feature_net.state_dict(), "feature_net.pth")
torch.save(successor_net.state_dict(), "successor_net.pth")
torch.save(intrinsic_reward_net.state_dict(), "intrinsic_reward_net.pth")
torch.save(w.detach().cpu(), "w.pth")
env.close()
