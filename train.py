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
import os

# Initialize Weights
def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
gamma = 0.99
lr = 2.5e-4
momentum = 0.95
epochs = 10000
feature_dim = 256
batch_size = 128
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999744
num_action_samples = 4
record_interval = 10
num_reward_updates = 5
num_successor_updates = 5

# Environment setup
video_folder = './videos'
os.makedirs(video_folder, exist_ok=True)

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_steps = 1000

# Initialize networks
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
successor_net = SuccessorNetwork(feature_dim, action_dim).to(device)
intrinsic_reward_net = IntrinsicRewardPredictor(feature_dim, state_dim).to(device)

# Initialize reward weight vector
w = nn.Parameter(torch.empty(feature_dim, device=device))
nn.init.kaiming_uniform_(w.unsqueeze(0), nonlinearity='relu')
w.requires_grad_()

# Target network
successor_net_prev = SuccessorNetwork(feature_dim, action_dim).to(device)
successor_net_prev.load_state_dict(successor_net.state_dict())
successor_net_prev.eval()

# Optimizers
optimizer_theta = optim.SGD(feature_net.parameters(), lr=lr, momentum=momentum)
optimizer_alpha = optim.SGD(successor_net.parameters(), lr=lr, momentum=momentum)
optimizer_theta_tilde = optim.SGD(intrinsic_reward_net.parameters(), lr=lr, momentum=momentum)
optimizer_w = optim.SGD([w], lr=lr, momentum=momentum)

# Replay Buffer
replay_buffer = ReplayBuffer(int(1e6))

# One-hot helper
def one_hot(actions, num_classes):
    return torch.eye(num_classes, device=actions.device)[actions]

# Training loop
for epoch in range(epochs):
    if epoch % 10 == 0:
        successor_net_prev.load_state_dict(successor_net.state_dict())

    if hasattr(env, 'close'):
        env.close()

    env = gym.make("LunarLander-v3")

    if epoch % record_interval == 0:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=f"epoch_{epoch}"
        )

    state, info = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    terminated, truncated = False, False
    epoch_l_r, epoch_l_a, epoch_loss_sr = [], [], []

    with tqdm(total=max_steps, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for step in range(max_steps):
            if terminated or truncated:
                break

            phi_s = feature_net(state)

            with torch.no_grad():
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                    phi_s_expanded = phi_s.expand(action_dim, -1)
                    m_s_a = successor_net_prev(phi_s_expanded, action_candidates)
                    q_values = (m_s_a @ w).squeeze(-1)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)

            replay_buffer.push(state, torch.tensor([action], device=device), reward, next_state, terminated)

            if len(replay_buffer) >= batch_size:
                for _ in range(num_reward_updates):
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                    batch_states = batch_states.squeeze(1)
                    phi_s_batch = feature_net(batch_states).squeeze(1)
                    phi_next_s_batch = feature_net(batch_next_states).squeeze(1)
                    batch_actions_oh = one_hot(batch_actions.squeeze(-1).long(), action_dim)

                    with torch.no_grad():
                        action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                        phi_next_s_batch_expanded = phi_next_s_batch.unsqueeze(1).expand(-1, action_dim, -1)
                        action_candidates_expanded = action_candidates.unsqueeze(0).expand(batch_size, -1, -1)
                        m_sDash_a = successor_net_prev(phi_next_s_batch_expanded, action_candidates_expanded)
                        q_values = (m_sDash_a @ w).squeeze(-1)
                        best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]

                    reward_pred_batch = (phi_s_batch @ w).unsqueeze(-1)
                    l_r = ((batch_rewards.squeeze(-1) - reward_pred_batch) ** 2).mean()
                    reconstructed_states = intrinsic_reward_net(phi_s_batch)
                    l_a = ((reconstructed_states - batch_states.squeeze(1)) ** 2).mean()

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

                    epoch_l_r.append(l_r.item())
                    epoch_l_a.append(l_a.item())

                for _ in range(num_successor_updates):
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                    batch_states = batch_states.squeeze(1)
                    phi_s_batch = feature_net(batch_states).squeeze(1)
                    phi_next_s_batch = feature_net(batch_next_states).squeeze(1)
                    batch_actions_oh = one_hot(batch_actions.squeeze(-1).long(), action_dim)

                    with torch.no_grad():
                        action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                        phi_next_s_batch_expanded = phi_next_s_batch.unsqueeze(1).expand(-1, action_dim, -1)
                        action_candidates_expanded = action_candidates.unsqueeze(0).expand(batch_size, -1, -1)
                        m_sDash_a = successor_net_prev(phi_next_s_batch_expanded, action_candidates_expanded)
                        q_values = (m_sDash_a @ w).squeeze(-1)
                        best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]

                    optimizer_alpha.zero_grad()
                    phi_s_batch_detached = phi_s_batch.detach()
                    best_m_sDash_a_detached = best_m_sDash_a.detach()
                    target_M = phi_s_batch_detached + gamma * best_m_sDash_a_detached * (1 - batch_dones)
                    m_sa_batch = successor_net(phi_s_batch_detached, batch_actions_oh)

                    loss_sr = ((target_M - m_sa_batch) ** 2).mean()

                    if torch.isnan(loss_sr) or torch.isinf(loss_sr):
                        print("❌ Skipping loss_sr due to instability")
                        continue

                    loss_sr.backward()
                    torch.nn.utils.clip_grad_norm_(successor_net.parameters(), max_norm=50)
                    optimizer_alpha.step()

                    epoch_loss_sr.append(loss_sr.item())

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

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Save models
torch.save(feature_net.state_dict(), "feature_net.pth")
torch.save(successor_net.state_dict(), "successor_net.pth")
torch.save(intrinsic_reward_net.state_dict(), "intrinsic_reward_net.pth")
torch.save(w.detach().cpu(), "w.pth")

env.close()
