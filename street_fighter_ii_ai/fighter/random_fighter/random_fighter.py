from street_fighter_ii_ai.fighter.fighter import Fighter
from numpy.random import default_rng

class RandomFighterSettings:
    rng=default_rng()
    action_space_size = 12

class RandomFighter(Fighter):
    def __init__(self, settings = RandomFighterSettings()):
        self.settings = settings

    def act(self, state):
        return self.settings.rng.integers(self.settings.action_space_size)

    def reset(self):
        return

    def cache(self, state, action, next_state, reward, done):
        return

    def save(self, save_path):
        return