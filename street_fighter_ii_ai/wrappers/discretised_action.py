from gymnasium import ActionWrapper#
from gymnasium.spaces import Discrete
import numpy as np

class DiscretisedAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(env.action_space.n)

    def action(self, action):
        action_binary = np.zeros(self.env.action_space.n)
        action_binary[action] = 1
        return action_binary
