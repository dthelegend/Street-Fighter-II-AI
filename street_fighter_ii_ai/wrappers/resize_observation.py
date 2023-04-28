from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tensorflow as tf
import numpy as np

class ResizeObservation(ObservationWrapper):

    def __init__(self, env, /, shape):
        super().__init__(env)

        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        observation = tf.image.resize(observation, self.shape, antialias=False)
        observation = observation / 255.0
        return observation #.squeeze(0)

