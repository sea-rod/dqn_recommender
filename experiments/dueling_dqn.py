# experiments/dueling_dqn.py
import torch.nn as nn
import torch


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU())

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
