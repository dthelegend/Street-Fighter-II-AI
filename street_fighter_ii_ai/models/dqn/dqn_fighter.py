from street_fighter_ii_ai.models.dqn.replay_memory import ReplayMemory
from street_fighter_ii_ai.models.dqn.dqn import DQN, DQNMode
from street_fighter_ii_ai.shared.abc.fighter import Fighter
import math
import random
import torch
import pathlib
import numpy as np

class DQNFighter(Fighter):
    def __init__(self, /, obs_shape, action_space, memory_size = 10000, eps_start = 0.9, eps_end = 0.05, eps_decay = 1000, state_dict = None, sync_every = 1e4, batch_size = 128) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(obs_shape, action_space.n).to(self.device)
        if(state_dict):
            self.model.load_state_dict(state_dict)
        self.memory = ReplayMemory(memory_size)
        self.action_space = action_space
        self.steps = 0
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.eps_threshold = lambda step : eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)
        self.sync_every = sync_every
        self.batch_size = batch_size

    def act(self, state):
        sample = random.random()

        self.steps += 1
        
        if sample > self.eps_threshold(self.steps):
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                device_state = state.to(self.device)
                return self.model(device_state).max(1)[1].view(1, 1)
        else:
            return self.action_space.sample()

    def reset(self, clear_memory=False):
        self.steps = 0
        if(clear_memory):
            self.memory.clear()

    def cache(self, state, action, next_state, reward, done):
        self.memory.push(state, action, None if done else next_state, reward)
    
    def learn(self):
        if (self.steps % self.sync_every == 0):
            self.sync_Q_target()
        
        if(len(self.memory) < self.batch_size):
            return None, None
        
        batch = self.memory.sample(self.batch_size)

        for state, next_state, action, reward in batch:
            td_est = self.td_estimate(state, action)
            td_tgt = self.td_target(reward, next_state)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state):
        if next_state is None:
            return reward
        next_state_Q = self.model(next_state, mode=DQNMode.ONLINE)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.model(next_state, mode=DQNMode.TARGET)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.model.target.load_state_dict(self.model.online.state_dict())

    def save(self, save_dir = pathlib.Path("./nets")):
        save_path = save_dir / f"DQN_net_{self.steps}.chkpt"
        torch.save(self.model.state_dict(), save_path)
