from street_fighter_ii_ai.fighter.fighter import Fighter
import numpy as np

class HumanFighter(Fighter):
    def __init__(self, env):
        self.env = env

    def act(self, state):
        x = input() # lrup

        match x:
            case "w":
                res = 1
            case "a":
                res = 2
            case "s":
                res = 3
            case "d":
                res = 4
            case "u":
                res = "A"
            case "i":
                res = "B"
            case "o":
                res = "C"
            case "j":
                res = "X"
            case "k":
                res = "Y"
            case "l":
                res = "Z"
            case "":
                res = 0

        return self.env._decode_discrete_action[[res]]


    def reset(self):
        pass

    def cache(self, state, action, next_state, reward, done):
        pass

    def save(self, save_path):
        pass
