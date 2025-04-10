# # import torch
# # import torch.nn as nn

# # class FeatureNetwork(nn.Module):
# #     def __init__(self, state_dim, feature_dim):
# #         super().__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(state_dim, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, feature_dim)
# #         )

# #     def forward(self, state):
# #         return self.fc(state)

# import torch
# import torch.nn as nn

# class FeatureNetwork(nn.Module):
#     def __init__(self, state_dim, feature_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.LayerNorm(256),  # LayerNorm added here
#             nn.ReLU(),
#             nn.Linear(256, feature_dim)
#         )

#     def forward(self, state):
#         return self.fc(state)

import torch
import torch.nn as nn

class FeatureNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.fc(state)
