from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import torchvision.transforms as T
import numpy as np

class GrayScaleObservation(ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[1:] # Chop off the channels
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = T.Grayscale()
        observation = transform(observation)
        return observation