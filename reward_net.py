# import torch
# import torch.nn as nn

# class RewardNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self.output = nn.Linear(128, 1)  # Final reward prediction layer

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.shared:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 nn.init.zeros_(m.bias)
#         nn.init.kaiming_uniform_(self.output.weight, nonlinearity='relu')
#         nn.init.zeros_(self.output.bias)

#     def forward(self, state):
#         features = self.shared(state)       # This is phi(s)
#         reward = self.output(features)      # This is r(s)
#         w = self.output.weight.T.detach()
#         return reward, features, w

# reward_net.py

import torch
import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.output = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state):
        features = self.shared(state)
        reward = self.output(features)
        w = self.output.weight.T.detach()
        return reward, features, w
