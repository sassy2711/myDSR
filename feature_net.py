import torch
import torch.nn as nn

class FeatureNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state):
        return self.fc(state)
