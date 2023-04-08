import torch
from torch import nn
import copy
from enum import Enum

class DQNMode(Enum):
    ONLINE="online"
    TARGET="target"

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, mode = DQNMode.TARGET):
        if mode == DQNMode.ONLINE:
            return self.online(input)
        return self.target(input)
