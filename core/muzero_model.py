import torch
import torch.nn as nn
import typing

class MuZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define your architecture here (ResNet, FC, etc.)
        # This is a dummy example to make the code run
        self.representation = nn.Sequential(
            nn.Linear(config.obs_shape, config.hidden_size), 
            nn.ReLU(), 
        )
        self.dynamics = nn.Sequential(
            nn.Linear(config.hidden_size + 1, config.hidden_size), 
            nn.ReLU()
        )
        self.prediction = nn.Sequential(
            nn.Linear(config.hidden_size, config.action_space_size), # Policy
            nn.Linear(config.hidden_size, 1) # Value
        )

    def prediction_step(self, state):
        # Used by MCTS
        pass

    def get_weights(self):
        # HELPER FOR RAY: Move to CPU before sending
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        # HELPER FOR RAY: Load weights
        self.load_state_dict(weights)