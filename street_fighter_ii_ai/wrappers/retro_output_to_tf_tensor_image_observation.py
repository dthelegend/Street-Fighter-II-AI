from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import tensorflow as tf

class RetroOutputToTFTensorObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = tf.convert_to_tensor(observation.copy(), dtype=tf.uint8)
        return observation
