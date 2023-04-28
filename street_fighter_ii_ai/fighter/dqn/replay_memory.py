from collections import namedtuple, deque
import numpy as np

class ReplayMemory():

    def __init__(self, capacity):
        self.states = np.zeros(shape=(capacity,), dtype=(np.float32, (84, 84, 1)))
        self.actions = np.zeros(shape=(capacity,), dtype=np.int8)
        self.next_states = np.zeros(shape=(capacity,), dtype=(np.float32, (84, 84, 1)))
        self.rewards = np.zeros(shape=(capacity,), dtype=np.float32)
        self.dones = np.zeros(shape=(capacity,), dtype=np.bool)

        self.current_size = 0
        self.index = 0
        self.max_size = capacity
    
    @property
    def data(self):
        return (self.states[:self.current_size], self.actions[:self.current_size], self.next_states[:self.current_size], self.rewards[:self.current_size], self.dones[:self.current_size])

    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.dones[indices])

    def push(self, *args):
        """Save a transition"""
        
        self.states[self.index], self.actions[self.index], self.next_states[self.index], self.rewards[self.index], self.dones[self.index] = args

        self.index = (self.index + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def clear(self):
        self.current_size = 0

    def __len__(self):
        return self.current_size