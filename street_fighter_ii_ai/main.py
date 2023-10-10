import os
import tensorflow as tf
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disable Tensorflow logging

from street_fighter_ii_ai.arena.training_arena import TrainingArena, TrainingMode
from street_fighter_ii_ai.fighter.dqn.dddqn_fighter import DDDQNFighter, DDDQNFighterSettings
from street_fighter_ii_ai.fighter.random_fighter.random_fighter import RandomFighter, RandomFighterSettings
import pathlib
import time
import csv

NUM_EPISODES = 10000
SAVE_EVERY = 10
RENDER = True

LOG_PATH = pathlib.Path("logs") / "dddqn_train_{}.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SAVE_PATH = pathlib.Path("nets") / "dddqn" / "dddqn_{}.ckpt"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

def run_random():
    with TrainingArena(TrainingMode.RYU) as arena:
        # Create a random fighter
        random_settings = RandomFighterSettings()
        random_settings.action_space_size = arena.env.action_space.n
        player = RandomFighter(random_settings)

        # Set the player
        arena.set_player(player)

        # Run the arena
        try:
            for _ in range(NUM_EPISODES):
                while arena.step() is not None:
                    pass
                arena.reset()
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print("Training interrupted by exception: " + str(e))

def run_dqn(num_episodes):
    # Create a DDDQN fighter
    player_settings = DDDQNFighterSettings()
    # player_settings.weights = SAVE_PATH
    # player_settings.action_space_size = arena.env.action_space.n
    # player_settings.observation_shape = arena.env.observation_space.shape
    player_settings.weights = tf.train.latest_checkpoint(SAVE_PATH.parent)
    player = DDDQNFighter(settings=player_settings)

    log_file = open(str(LOG_PATH).format(time.strftime("%Y%m%d-%H%M%S")), "wt") if LOG_PATH is not None else None
    csv_writer = None

    try:
        while num_episodes > 0:
            mode = random.choice([TrainingMode.RYU, TrainingMode.KEN, TrainingMode.CHUN_LI, TrainingMode.HONDA])
            random_length = min(random.randint(1, 10), num_episodes)
            num_episodes -= random_length
            with TrainingArena(mode) as arena:
                for current_episode in range(random_length):
                    # Set the player
                    arena.set_player(player)

                    while arena.step() is not None:
                        pass

                    if(log_file is not None):
                        if player.metrics is not None and arena.info is not None:
                            if csv_writer is None:
                                print(f"Metrics names: {player.metrics.keys()}")
                                print(f"Arena Info keys: {arena.info.keys()}")
                                csv_writer = csv.DictWriter(log_file, fieldnames=("episode", *player.metrics.keys(), *arena.info.keys()))
                                csv_writer.writeheader()
                            
                            print("Writing to log file...")
                            csv_writer.writerow({"episode": current_episode, **player.metrics, **arena.info})
                            log_file.flush()
                            print("Written to log file")

                    arena.reset()

                    # Save the player
                    if current_episode % SAVE_EVERY == 0:
                        player.save(str(SAVE_PATH).format(time.strftime("%Y%m%d-%H%M%S")))
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print("Training interrupted by exception: " + str(e))
    finally:
        player.save(str(SAVE_PATH).format(time.strftime("%Y%m%d-%H%M%S")))
        if(log_file is not None):
            log_file.close()

def main():
    # run_random()
    run_dqn(NUM_EPISODES)

if __name__=="__main__":
    main()