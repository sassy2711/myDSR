import torch
import torch.nn as nn

class SuccessorNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(SuccessorNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, 256)  # (256 + 2 = 258)
        self.fc2 = nn.Linear(256, feature_dim)  # Output must match feature_dim

    def forward(self, phi_s, action):
        if action.dim() == 1:  # If action is (batch_size,), reshape it
            action = action.unsqueeze(-1)  # Convert to (batch_size, 1)
        elif action.dim() == 2 and action.shape[1] != 2:  # Ensure (batch_size, 2)
            action = action.view(-1, 2)

        x = torch.cat([phi_s, action], dim=-1)  # phi_s: (batch_size, 256), action: (batch_size, 2)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output shape: (batch_size, 256)
