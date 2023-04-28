from street_fighter_ii_ai.fighter.dqn.replay_memory import ReplayMemory
from street_fighter_ii_ai.fighter.dqn.models import DoubleDuelingDeepQNetwork as DDDQN
from street_fighter_ii_ai.fighter.fighter import Fighter
import tensorflow as tf
import pathlib
import time
import numpy as np

class DDDQNFighterSettings:
    learning_rate = 0.00025
    batch_size = 32

    action_space_size = 12

    weights = None

    exploration_rate_decay = 0.99999975
    exploration_rate_min = 0.1

    burnin = 1e4
    sync_every = 1e4
    learn_every = 3

    memory_size = int(1e5)

class DDDQNFighter(Fighter):
    def __init__(self, settings=DDDQNFighterSettings()):
        self.settings = settings

        self.model = DDDQN(self.settings.action_space_size)

        if self.settings.weights is not None:
            self.model.load_weights(self.settings.weights)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings.learning_rate), loss=tf.keras.losses.Huber(), jit_compile=True)

        self.exploration_rate = 1.0

        self.memory = ReplayMemory(self.settings.memory_size)

        self.sync_counter = 0
        self.learn_counter = 0
        self.burnin = self.settings.burnin

    def act(self, state):
        
        if np.random.rand() >= self.exploration_rate:
            action_values = self.model(tf.expand_dims(state, axis=0))
            output = tf.math.argmax(tf.squeeze(action_values, axis=0)).numpy()
        else:
            output = np.random.randint(self.settings.action_space_size)
        
        self.exploration_rate *= self.settings.exploration_rate_decay
        self.exploration_rate = max(self.settings.exploration_rate_min, self.exploration_rate)

        if self.sync_counter > self.settings.sync_every:
            print("Syncing target network")
            self.sync_Q_target()
            self.sync_counter = 0
        else:
            self.sync_counter += 1
        
        if self.burnin > 0:
            self.burnin -= 1
        elif self.learn_counter > self.settings.learn_every:
            self.learn()
            self.learn_counter = 0
        else:
            self.learn_counter += 1

        return output

    def reset(self, clear_memory=False):
        self.sync_counter = 0
        self.learn_counter = 0
        self.burnin = self.settings.burnin
        if(clear_memory):
            self.memory.clear()

    def cache(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def learn(self):
        if(len(self.memory) < self.settings.batch_size):
            return

        self.model.train_on_batch(self.memory.sample(self.settings.batch_size))

    def save(self, save_path):
        self.model.save(save_path, save_format="tf")
        print("Saved model to {}".format(save_path))