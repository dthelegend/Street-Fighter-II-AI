import retro
from street_fighter_ii_ai.shared.wrappers.grey_scale_observation import GrayScaleObservation
from street_fighter_ii_ai.shared.wrappers.retro_output_to_torch_image_observation import RetroOutputToTorchImageObservation
from street_fighter_ii_ai.shared.wrappers.resize_observation import ResizeObservation
from enum import IntEnum
from street_fighter_ii_ai.shared.abc.fighter import Fighter

class ArenaMode(IntEnum):
    TRAIN = 1,
    FIGHT = 0

class Arena():
    def __init__(self, mode: ArenaMode, game_name="StreetFighterIISpecialChampionEdition-Genesis"):
        self.env = retro.make(game=game_name)

        self.env = RetroOutputToTorchImageObservation(self.env)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        
        self.mode = mode
        self.reset()

    def set_player1(self, player1: Fighter):
        self.player1 = player1
    
    def set_player2(self, player2: Fighter):
        self.player2 = player2
    
    @staticmethod
    def enter(mode, player1, player2 = None):
        arena = Arena(mode)
        arena.set_player1(player1)
        if(player2 is not None):
            arena.set_player2(player2)
        return arena
    
    def leave(self):
        self.env.close()

    def reset(self):
        self.state, _ = self.env.reset()

    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.leave()
    
    def save(self):
        self.player1.save()

    def step(self, /, render=True):
        action = self.player1.act(self.state)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        if(self.mode == ArenaMode.TRAIN):
            self.player1.cache(self.state, action, next_state, reward, terminated or truncated)
            self.player1.learn()
        
        self.state = next_state

        if(render): 
            self.env.render()
        
        if terminated or truncated:
            return None

        return self.state

