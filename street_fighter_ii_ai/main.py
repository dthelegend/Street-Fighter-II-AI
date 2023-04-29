import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disable Tensorflow logging

from street_fighter_ii_ai.arena.training_arena import TrainingArena, TrainingMode
from street_fighter_ii_ai.fighter.dqn.dddqn_fighter import DDDQNFighter
import pathlib

NUM_EPISODES = 10000
SAVE_EVERY = 10

SAVE_PATH = pathlib.Path("nets") / "dddqn.ckpt"

def main():
    player = DDDQNFighter()

    if SAVE_PATH.exists():
        player.load(SAVE_PATH)

    with TrainingArena(TrainingMode.RYU, player) as arena:
        try:
            for current_episode in range(NUM_EPISODES):
                while arena.step() is not None:
                    pass
                arena.reset()
                if current_episode % SAVE_EVERY == 0:
                    player.save(SAVE_PATH)
        except (KeyboardInterrupt, Exception) as e:
            player.save(SAVE_PATH)
            if not isinstance(e, KeyboardInterrupt):
                raise e
            print("Training interrupted by user")

if __name__=="__main__":
    main()