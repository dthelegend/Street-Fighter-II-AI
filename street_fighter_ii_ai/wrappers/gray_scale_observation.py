from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tensorflow as tf
import numpy as np

class GrayScaleObservation(ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = (1,) + self.observation_space.shape[1:] # add a dimension for grayscale
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = tf.image.rgb_to_grayscale(observation)

        return observation