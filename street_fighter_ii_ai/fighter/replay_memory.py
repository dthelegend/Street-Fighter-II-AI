from typing import Tuple
import numpy as np

class PrioritisedExperienceReplayMemory():

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 0.01):
        self.states = np.zeros(shape=(capacity,), dtype=(np.float32, (84, 84, 1)))
        self.actions = np.zeros(shape=(capacity,), dtype=np.int8)
        self.next_states = np.zeros(shape=(capacity,), dtype=(np.float32, (84, 84, 1)))
        self.rewards = np.zeros(shape=(capacity,), dtype=np.float32)
        self.dones = np.zeros(shape=(capacity,), dtype=np.bool)

        self.alpha_errors = np.zeros(shape=(capacity,), dtype=np.float64)

        self.current_size = 0
        self.index = 0
        self.max_size = capacity
        self.alpha = alpha
        self.epsilon = epsilon
    
    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.states[:self.current_size], self.actions[:self.current_size], self.next_states[:self.current_size], self.rewards[:self.current_size], self.dones[:self.current_size])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.current_size < batch_size:
            raise Exception("Not enough data in the memory to sample a batch of size {}".format(batch_size))

        indices = np.random.choice(self.current_size, batch_size, p = self.alpha_errors[:self.current_size] / sum(self.alpha_errors[:self.current_size]), replace=False)
        return (self.states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.dones[indices])

    def push(self, *args):
        td_error, self.states[self.index], self.actions[self.index], self.next_states[self.index], self.rewards[self.index], self.dones[self.index] = args
        self.alpha_errors[self.index] = (td_error + self.epsilon) ** self.alpha

        self.index = (self.index + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def clear(self):
        self.current_size = 0

    def __len__(self):
        return self.current_size