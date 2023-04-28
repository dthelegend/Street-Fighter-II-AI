from abc import ABC, abstractmethod
import pathlib

class Fighter(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def cache(self, state, action, next_state, reward, done):
        raise NotImplementedError()

    @abstractmethod
    def save(self, save_path):
        raise NotImplementedError()