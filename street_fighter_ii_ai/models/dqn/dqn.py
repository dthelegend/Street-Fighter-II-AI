import torch
from torch import nn
import copy
from enum import Enum

class DQNModel(Enum):
    ONLINE="online"
    TARGET="target"

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model = DQNModel.TARGET):
        # Check if the tensor has a batch dimension and add one if it doesn't
        if input.ndim == 3:
            input = input.unsqueeze(0)
        if input.ndim != 4:
            raise ValueError()

        if model == DQNModel.ONLINE:
            return self.online(input)
        return self.target(input)
