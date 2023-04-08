from abc import ABC, abstractmethod

class Fighter(ABC):

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
    def save(self):
        raise NotImplementedError()