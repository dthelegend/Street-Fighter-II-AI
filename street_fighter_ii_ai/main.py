from street_fighter_ii_ai.shared.arena import Arena, ArenaMode
from street_fighter_ii_ai.models.dqn.dqn_fighter import DQNFighter
import torch

def main():
    arena = Arena(mode=ArenaMode.TRAIN)

    arena.set_player1(DQNFighter(obs_shape=arena.state.shape, action_space=arena.env.action_space))
        
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    while num_episodes > 0:
        if arena.step() is None:
            arena.reset()
            num_episodes -= 1
            arena.save()

if __name__=="__main__":
    main()