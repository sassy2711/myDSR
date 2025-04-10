import torch.nn as nn

class IntrinsicRewardPredictor(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_dim=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, phi_s):
        # phi_s: (batch_size, feature_dim)
        reconstructed_state = self.decoder(phi_s)
        return reconstructed_state  # shape: (batch_size, state_dim)
