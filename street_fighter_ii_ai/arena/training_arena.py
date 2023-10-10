from enum import Enum
from street_fighter_ii_ai.fighter.fighter import Fighter
from street_fighter_ii_ai.arena.arena import Arena
from street_fighter_ii_ai.utils import create_environment, ROOT_DIR

GAME_NAME="StreetFighterIISpecialChampionEdition-Genesis"
STATES_DIR = ROOT_DIR / "custom_integrations" / GAME_NAME

class TrainingMode(Enum):
    BLANKA = "blanka"
    CHUN_LI = "Champion.Level1.ChunLiVsGuile.state"
    DHALSIM = "dhalsim"
    HONDA = "Champion.Level1.HondaVsChunLi.state"
    GUILE = "Guile"
    KEN = "Champion.Level1.KenVsRyu.state"
    RYU = "Champion.Level1.RyuVsGuile.state"
    ZANGIEF = "zangief"
    BALROG = "balrog"
    BISON = "bison"
    SAGAT = "sagat"
    VEGA = "vega"
    DEFAULT = RYU

class TrainingArena(Arena):
    def __init__(self, mode: TrainingMode = TrainingMode.DEFAULT, log_info=True):
        self.env = None
        self.mode = mode
        self.state = None
        self.info = None
        self.player = None
    
    def reset(self):
        self.state, self.info = self.env.reset()
    
    def set_player(self, player: Fighter):
        self.player = player
    
    def __enter__(self):
        if self.mode == None:
            self.env = create_environment()
        else:
            self.env = create_environment(str(STATES_DIR / self.mode.value))
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()
        self.env = None

    def step(self):
        if(self.env is None):
            raise ValueError("No environment initialised. Please use in Context Manager")

        action = self.player.act(self.state)

        next_state, reward, terminated, truncated, self.info = self.env.step(action)

        self.player.cache(self.state, action, next_state, reward, terminated or truncated)
        
        self.state = next_state
        
        if terminated or truncated:
            return None

        return self.state

    def render(self): 
        self.env.render()
