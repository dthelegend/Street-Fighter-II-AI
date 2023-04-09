from street_fighter_ii_ai.models.dqn.replay_memory import ReplayMemory
from street_fighter_ii_ai.models.dqn.dqn import DQN, DQNModel
from street_fighter_ii_ai.shared.abc.fighter import Fighter
import math
import random
import torch
import pathlib
import numpy as np

class DQNFighter(Fighter):
    def __init__(self, /, obs_shape, action_space, memory_size = 10000, eps_start = 0.9, eps_end = 0.05, eps_decay = 1000, state_dict = None, sync_every = 1e4, batch_size = 128, burnin = 1e4) -> None:
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
        self.burnin = burnin

    def act(self, state):
        sample = random.random()
        
        if sample > self.eps_threshold(self.steps):
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                device_state = state.to(self.device)

                action_values = self.model(device_state, model = DQNModel.ONLINE)
                action_idx = torch.argmax(action_values, axis=1).item()

                output = np.zeros(12, dtype=bool)
                output[action_idx] = 1
        else:
            output = np.array(self.action_space.sample(), dtype=bool)
        
        self.steps += 1

        return output

    def reset(self, clear_memory=False):
        self.steps = 0
        if(clear_memory):
            self.memory.clear()

    def cache(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def learn(self):
        if (self.steps % self.sync_every == 0):
            self.sync_Q_target()
        
        if (self.steps < self.burnin):
            return None, None
        
        if(len(self.memory) < self.batch_size):
            return None, None
        
        state, action, next_state, reward, done = zip(*self.memory.sample(self.batch_size))

        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.from_numpy(np.array(action)).to(self.device)

        action

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):

        print(action.shape)

        current_Q = self.model(state, model=DQNModel.ONLINE)[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        if done:
            return reward
        next_state_Q = self.model(next_state, model=DQNModel.ONLINE)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.model(next_state, model=DQNModel.TARGET)[
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
