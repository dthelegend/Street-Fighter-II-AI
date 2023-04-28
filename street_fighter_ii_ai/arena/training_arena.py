import retro
from enum import Enum
from street_fighter_ii_ai.fighter.fighter import Fighter
from street_fighter_ii_ai.arena.arena import Arena
from street_fighter_ii_ai.utils import create_environment

GAME_NAME="StreetFighterIISpecialChampionEdition-Genesis"

class TrainingMode(Enum):
    DEFAULT = retro.State.DEFAULT
    BLANKA = "blanka"
    CHUN_LI = "chun-li"
    DHALSIM = "dhalsim"
    HONDA = "honda"
    GUILE = "Guile"
    KEN = "Champion.Level1.KenVsHonda.state"
    RYU = "Champion.Level1.RyuVsGuile.state"
    ZANGIEF = "zangief"
    BALROG = "balrog"
    BISON = "bison"
    SAGAT = "sagat"
    VEGA = "vega"

class TrainingArena(Arena):
    def __init__(self, mode: TrainingMode = TrainingMode.DEFAULT, player: Fighter = None):
        self.env = None
        self.mode = mode
        self.state = None
        self.info = None
        self.player = player
    
    def reset(self):
        self.state, self.info = self.env.reset()
    
    def __enter__(self):
        self.env = create_environment(self.mode.value)
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()
        self.env = None

    def step(self, /, render=True):
        if(self.env is None):
            raise ValueError("No environment initialised. Please use in Context Manager")

        action = self.player.act(self.state)

        next_state, reward, terminated, truncated, self.info = self.env.step(action)

        self.player.cache(self.state, action, next_state, reward, terminated or truncated)
        
        self.state = next_state

        if(render): 
            self.env.render()
        
        if terminated or truncated:
            return None

        return self.state
