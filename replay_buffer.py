import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Fixed-size buffer
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors
        states = torch.stack(states)  # Ensure batch dimension consistency
        actions = torch.stack(actions)  # Ensure (batch_size, action_dim)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
