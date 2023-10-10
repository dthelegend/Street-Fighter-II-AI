import gymnasium as gym
import numpy as np

class StreetFighterDiscretisedAction(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Use Street Fighter-specific discrete actions
    """

    def __init__(self, env, /, combos_set_one=[['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['UP', 'RIGHT'], ['DOWN', 'RIGHT'], ['LEFT', 'DOWN'], ['LEFT', 'UP'], ['RIGHT', 'DOWN']], combo_set_two=[['A'], ['B'], ['C'], ['X'], ['Y'], ['Z']]):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo_one, combo_two in ((x, y) for x in [[], *combos_set_one] for y in [[], *combo_set_two]):
            combo = combo_one + combo_two
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()