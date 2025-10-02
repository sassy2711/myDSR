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
#from reward_net import RewardNetwork


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
gamma = 0.99
epochs = 2500
feature_dim = 128
num_action_samples = 4  # All possible discrete actions
batch_size = 64
epsilon_decay_steps = 2000  # number of epochs over which to linearly decay

# Epsilon-Greedy Parameters
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.997

# Learning Rates
lr_theta = 1e-3
lr_alpha = 5e-5
lr_tilde = 1e-4
lr_w = 5e-5

# Load environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_steps = 800
hard_update_interval = 10
# Networks
feature_net = FeatureNetwork(state_dim, feature_dim).to(device)
#reward_net = RewardNetwork(state_dim).to(device)
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
#optimizer_reward = optim.SGD(reward_net.parameters(), lr=lr_theta, momentum=0.95)
optimizer_alpha = optim.SGD(successor_net.parameters(), lr=lr_alpha, momentum=0.95)
optimizer_theta_tilde = optim.SGD(intrinsic_reward_net.parameters(), lr=lr_tilde, momentum=0.95)
optimizer_w = optim.SGD([w], lr=lr_w, momentum=0.95)
# optimizer_reward = optim.Adam(reward_net.parameters(), lr=lr_theta)
# optimizer_alpha = optim.Adam(successor_net.parameters(), lr=lr_alpha)

# Replay Buffer
buffer_capacity = 10000
replay_buffer = ReplayBuffer(buffer_capacity)

# One-hot encoding
def one_hot(actions, num_classes):
    return torch.eye(num_classes, device=actions.device)[actions]

def soft_update(target_net, source_net, tau=0.01):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def check_requires_grad(net, name=""):
    for n, p in net.named_parameters():
        if p.requires_grad and p.grad is not None:
            print(f"✅ Grad OK in {name}: {n}")
        elif p.requires_grad and p.grad is None:
            print(f"⚠️ No grad for {name}: {n}")
        elif not p.requires_grad:
            print(f"⛔ {name} param not requiring grad: {n}")


#print(w.shape)
# Training loop
for epoch in range(epochs):
    if hasattr(env, 'close'):
        env.close()

    if epoch % record_interval == 0:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=f"epoch_{epoch}"
        )
    else:
        env = gym.make("CartPole-v1")

    epoch_l_r = []
    epoch_l_a = []
    epoch_loss_sr = []

    #soft_update(successor_net_prev, successor_net, tau=0.01)
    # Perform soft update at each step of the training loop
    if epoch % hard_update_interval == 0:
        successor_net_prev.load_state_dict(successor_net.state_dict())  # Hard update

    state, _ = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    terminated, truncated = False, False

    with tqdm(total=max_steps, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for step in range(max_steps):
            if terminated or truncated:
                break

            #r_s, phi_s, w = reward_net(state)
            phi_s = feature_net(state)
            phi_s = phi_s.detach()
            w = w.detach()  # ⛔ Detach after forward to avoid implicit autograd tracking

            # Epsilon-greedy action selection
            # with torch.no_grad():
                #print("Epsilon: ", epsilon)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                phi_s_exp = phi_s.expand(action_dim, -1)
                m_s_a = successor_net(phi_s_exp, action_candidates)
                m_s_a = m_s_a.detach()
                q_values = (m_s_a @ w).squeeze(-1)
                q_values = q_values.detach()
                action = q_values.argmax().item()
                #action = action.detach()
                    # print(m_s_a.shape)
                    # print(q_values.shape)
                    # print(action_candidates.shape)
                    # print(phi_s_exp.shape)
                    # print(phi_s.shape)
                    #print(action.shape)

            next_state, reward, terminated, truncated, _ = env.step(action)
            #next_state = next_state.detach()
            #reward = reward.detach()
            #terminated = terminated.detach()
            #truncated = truncated.detach()
            #reward /= 10.0  # Reward scaling

            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).unsqueeze(0)
            #reward = reward  # Convert to plain float
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            replay_buffer.push(state, torch.tensor([action], device=device), reward, next_state, terminated)


            # ----- Training step -----
            if len(replay_buffer) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                batch_states = batch_states.squeeze(1)
                batch_next_states = batch_next_states.squeeze(1)
                phi_s_batch = feature_net(batch_states)
                reconstructed_states = intrinsic_reward_net(phi_s_batch)
                #print(batch_states.shape)
                # ======== REWARD PHASE ========
                #reward_pred_batch, _, _ = reward_net(batch_states)  # Only use reward prediction here
                reward_pred_batch = phi_s_batch @ w
                #print(batch_rewards.shape)
                l_r = ((batch_rewards - reward_pred_batch) ** 2).mean()
                l_a = ((reconstructed_states - batch_states) ** 2).mean()
                epoch_l_r.append(l_r.item())
                epoch_l_a.append(l_a.item())
                reward_loss = l_r + l_a

                if torch.isnan(reward_loss) or torch.isinf(reward_loss):
                    print("❌ Skipping reward_loss due to instability")
                else:
                    optimizer_theta.zero_grad()
                    optimizer_alpha.zero_grad()
                    optimizer_theta_tilde.zero_grad()
                    optimizer_w.zero_grad()
                    reward_loss.backward()
                    torch.nn.utils.clip_grad_norm_(intrinsic_reward_net.parameters(), max_norm=50)
                    torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=50)
                    torch.nn.utils.clip_grad_norm_(w, max_norm=50)
                    optimizer_theta.step()
                    optimizer_theta_tilde.step()
                    optimizer_w.step()
                    phi_s_batch = phi_s_batch.detach()
                    reconstructed_states = reconstructed_states.detach()
                    reward_pred_batch = reward_pred_batch.detach()
                    # phi_s = phi_s.detach()
                    # reward_loss = reward_loss.detach()
                    # # 🧪 Debug: Check gradients
                    # print("\n🔍 Checking gradients after reward_loss.backward():")
                    # for name, param in reward_net.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"✅ reward_net param '{name}' has grad with mean: {param.grad.abs().mean():.6f}")
                    #     else:
                    #         print(f"⚠️ reward_net param '{name}' has NO grad!")

                    # for name, param in successor_net.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"❌ WARNING: successor_net param '{name}' has grad after reward_loss.backward()! Mean: {param.grad.abs().mean():.6f}")
                    #reward_pred_batch = reward_pred_batch.detach()  # Ensure it's not involved in any unwanted autograd

                    #check_requires_grad(reward_net, "RewardNet")

                phi_s_batch = feature_net(batch_states)
                phi_centered = phi_s_batch - phi_s_batch.mean(dim=0, keepdim=True)
                cov = (phi_centered.T @ phi_centered) / (phi_centered.size(0) - 1)

                I = torch.eye(cov.size(0), device=cov.device)
                l_d = ((cov - I) ** 2).sum()
                lambda_decor = 1e-3
                decor_loss = l_d*lambda_decor

                if torch.isnan(decor_loss) or torch.isinf(decor_loss):
                    print("❌ Skipping reward_loss due to instability")
                else:
                    optimizer_theta.zero_grad()
                    optimizer_alpha.zero_grad()
                    optimizer_theta_tilde.zero_grad()
                    optimizer_w.zero_grad()
                    decor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(intrinsic_reward_net.parameters(), max_norm=50)
                    torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=50)
                    #torch.nn.utils.clip_grad_norm_(w, max_norm=50)
                    optimizer_theta.step()
                    optimizer_theta_tilde.step()
                    phi_s_batch = phi_s_batch.detach()
                    # optimizer_w.step()
                    
                # ======== SR PHASE (completely detached from reward_net) ========
                with torch.no_grad():
                    phi_s_batch = feature_net(batch_states)
                    phi_next_s_batch = feature_net(batch_next_states)
                # print(phi_s_batch.shape)
                # print(phi_next_s_batch.shape)
                # print(w.shape)
                # ⛔ Detach reward outputs to sever gradient flow
                phi_s_batch = phi_s_batch.detach()
                phi_next_s_batch = phi_next_s_batch.detach()
                w = w.detach()

                batch_actions_oh = one_hot(batch_actions.squeeze(-1).long(), action_dim)
                batch_actions_oh = batch_actions_oh.detach()
                batch_dones = batch_dones.float().detach()

                # Compute target successor representation
                with torch.no_grad():
                    action_candidates = one_hot(torch.arange(action_dim, device=device), action_dim).float()
                    phi_next_exp = phi_next_s_batch.unsqueeze(1).expand(-1, action_dim, -1)
                    action_exp = action_candidates.unsqueeze(0).expand(batch_size, -1, -1)
                    B, A, F = phi_next_exp.shape
                    phi_next_exp_flat = phi_next_exp.reshape(B * A, F)
                    action_exp_flat = action_exp.reshape(B * A, action_dim)
                    m_sDash_a_flat = successor_net_prev(phi_next_exp_flat, action_exp_flat)
                    m_sDash_a = m_sDash_a_flat.view(B, A, F)
                    q_values = (m_sDash_a @ w).squeeze(-1)
                    # print(m_sDash_a.shape)
                    # print(q_values.shape)
                    best_m_sDash_a = m_sDash_a[torch.arange(batch_size), q_values.argmax(dim=1)]

                target_M = phi_s_batch + gamma * best_m_sDash_a * (1 - batch_dones)

                m_sa_batch = successor_net(phi_s_batch, batch_actions_oh)
                loss_sr = ((target_M - m_sa_batch) ** 2).mean()
                epoch_loss_sr.append(loss_sr.item())

                if torch.isnan(loss_sr) or torch.isinf(loss_sr):
                    print("❌ Skipping loss_sr due to instability")
                else:
                    optimizer_alpha.zero_grad()
                    optimizer_theta.zero_grad()
                    optimizer_theta_tilde.zero_grad()
                    optimizer_w.zero_grad()
                    loss_sr.backward()
                    # # 🧪 Debug: Check gradients again
                    # print("\n🔍 Checking gradients after loss_sr.backward():")
                    # for name, param in successor_net.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"✅ successor_net param '{name}' has grad with mean: {param.grad.abs().mean():.6f}")
                    #     else:
                    #         print(f"⚠️ successor_net param '{name}' has NO grad!")

                    # for name, param in reward_net.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"❌ WARNING: reward_net param '{name}' still has grad after loss_sr.backward()! Mean: {param.grad.abs().mean():.6f}")
                    #check_requires_grad(successor_net, "SuccessorNet")

                    torch.nn.utils.clip_grad_norm_(successor_net.parameters(), max_norm=50)
                    optimizer_alpha.step()

            state = next_state
            total_reward += reward   # Scale reward back for logging

            if step % 10 == 0:
                pbar.set_postfix(Total_Reward=total_reward)
            pbar.update(1)

    avg_l_r = np.mean(epoch_l_r) if epoch_l_r else float('nan')
    avg_l_a = np.mean(epoch_l_a) if epoch_l_a else float('nan')
    avg_loss_sr = np.mean(epoch_loss_sr) if epoch_loss_sr else float('nan')

    print(f"✅ Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.2f} | "
          f"Loss_r: {avg_l_r:.4f} | "f"Loss_a: {avg_l_a:.4f} | Loss_SR: {avg_loss_sr:.4f}")
    #print(f"🔁 Steps this epoch: {step + 1}")

    #epsilon = max(epsilon_min, epsilon * epsilon_decay)
    epsilon = max(epsilon_min, 1.0 - (epoch / epsilon_decay_steps) * (1.0 - epsilon_min))


# Save models
torch.save(feature_net.state_dict(), "feature_net.pth")
#torch.save(reward_net.state_dict(), "reward_net.pth")
torch.save(successor_net.state_dict(), "successor_net.pth")
torch.save(intrinsic_reward_net.state_dict(), "intrinsic_reward_net.pth")
torch.save(w.detach().cpu(), "w.pth")
env.close()




