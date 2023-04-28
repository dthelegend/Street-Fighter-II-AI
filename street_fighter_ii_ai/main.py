from street_fighter_ii_ai.arena.training_arena import TrainingArena, TrainingMode
from street_fighter_ii_ai.fighter.dqn.dddqn_fighter import DDDQNFighter
import pathlib

NUM_EPISODES = 1000
SAVE_EVERY = 100

SAVE_PATH = pathlib.Path("nets") / "dddqn.tf"

def main():
    player = DDDQNFighter()

    with TrainingArena(TrainingMode.RYU, player) as arena:
        try:
            for current_episode in range(NUM_EPISODES):
                while arena.step() is not None:
                    pass
                arena.reset()
                if current_episode % SAVE_EVERY == 0:
                    player.save(SAVE_PATH)
        except KeyboardInterrupt:
            player.save(SAVE_PATH)
            exit(0)

if __name__=="__main__":
    main()