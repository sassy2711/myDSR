# # import torch
# # import torch.nn as nn

# # # class SuccessorNetwork(nn.Module):
# # #     def __init__(self, feature_dim, action_dim):
# # #         super(SuccessorNetwork, self).__init__()
# # #         self.fc1 = nn.Linear(feature_dim + action_dim, 256)  # (256 + 2 = 258)
# # #         self.fc2 = nn.Linear(256, feature_dim)  # Output must match feature_dim

# # #     def forward(self, phi_s, action):
# # #         if action.dim() == 1:  # If action is (batch_size,), reshape it
# # #             action = action.unsqueeze(-1)  # Convert to (batch_size, 1)
# # #         elif action.dim() == 2 and action.shape[1] != 2:  # Ensure (batch_size, 2)
# # #             action = action.view(-1, 2)

# # #         x = torch.cat([phi_s, action], dim=-1)  # phi_s: (batch_size, 256), action: (batch_size, 2)
# # #         x = torch.relu(self.fc1(x))
# # #         return self.fc2(x)  # Output shape: (batch_size, 256)

# # class SuccessorNetwork(nn.Module):
# #     def __init__(self, feature_dim, action_dim):
# #         super(SuccessorNetwork, self).__init__()
# #         self.fc1 = nn.Linear(feature_dim + action_dim, 256)
# #         self.ln1 = nn.LayerNorm(256)  # LayerNorm after first layer
# #         self.fc2 = nn.Linear(256, feature_dim)

# #     def forward(self, phi_s, action):
# #         if action.dim() == 1:
# #             action = action.unsqueeze(-1)
# #         elif action.dim() == 2 and action.shape[1] != 2:
# #             action = action.view(-1, 2)

# #         x = torch.cat([phi_s, action], dim=-1)
# #         x = torch.relu(self.ln1(self.fc1(x)))
# #         return self.fc2(x)

# import torch
# import torch.nn as nn

# class SuccessorNetwork(nn.Module):
#     def __init__(self, feature_dim, action_dim):
#         super(SuccessorNetwork, self).__init__()
#         self.fc1 = nn.Linear(feature_dim + action_dim, 256)
#         self.fc2 = nn.Linear(256, feature_dim)
        
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 nn.init.zeros_(m.bias)

#     def forward(self, phi_s, action):
#         if action.dim() == 1:
#             action = action.unsqueeze(-1)
#         elif action.dim() == 2 and action.shape[1] != 2:
#             action = action.view(-1, 2)

#         x = torch.cat([phi_s, action], dim=-1)
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)

import torch
import torch.nn as nn

class SuccessorNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(SuccessorNetwork, self).__init__()
        # The input to this network will be the feature_dim (phi_s) + one-hot encoded action (action_dim)
        self.fc1 = nn.Linear(feature_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, feature_dim)  # Output dimension is same as feature_dim
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, phi_s, action):
        """
        Forward pass through the network.
        
        :param phi_s: Feature representation of the state (tensor of shape [batch_size, feature_dim])
        :param action: One-hot encoded action (tensor of shape [batch_size, action_dim])
        :return: Successor representation (tensor of shape [batch_size, feature_dim])
        """
        # Concatenate the state feature and one-hot encoded action
        x = torch.cat([phi_s, action], dim=-1)  # Concatenate along the last dimension
        x = torch.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        return self.fc2(x)  # Output the transformed feature representation
