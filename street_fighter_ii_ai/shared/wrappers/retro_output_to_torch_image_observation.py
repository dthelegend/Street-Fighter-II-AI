from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import torch

class RetroOutputToTorchImageObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[2:] + self.observation_space.shape[:2] # Get (C, H, W) from (H, W, C)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1)) # Convert (H, W, C) to (C, H, W)
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
