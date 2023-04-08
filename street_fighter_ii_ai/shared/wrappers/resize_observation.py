from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import torchvision.transforms as T

class ResizeObservation(ObservationWrapper):

    def __init__(self, env, /, shape):
        super().__init__(env)

        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape)

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])

        observation = transforms(observation)
        return observation #.squeeze(0)

