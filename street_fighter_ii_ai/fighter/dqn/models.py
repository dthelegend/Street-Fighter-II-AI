import tensorflow as tf
from enum import IntEnum

class DuelingDeepQNetwork(tf.keras.Model):
    def __init__(self, observation_shape: tuple, num_actions: int) -> None:
        super().__init__()

        self.net = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=observation_shape),
            tf.keras.layers.Conv2D(32, 8, 4, activation="relu"),
            tf.keras.layers.Conv2D(64, 4, 2, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu")
        ])

        self.V = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1)
        ])

        self.A = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_actions)
        ])

    @tf.function
    def call(self, input_tensor):
        x = self.net(input_tensor)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.math.reduce_mean(A, axis=1, keepdims=True))
        return Q

class DoubleDuelingDeepQNetwork(tf.keras.Model):
    ONLINE = 0
    TARGET = 1

    def __init__(self, observation_shape: tuple, num_actions: int, gamma = 0.9) -> None:
        super().__init__()

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.online = DuelingDeepQNetwork(observation_shape, num_actions)
        self.target = DuelingDeepQNetwork(observation_shape, num_actions)

    @tf.function
    def call(self, input_tensor, network = None):
        if network == self.TARGET:
            return self.target(input_tensor)
        return self.online(input_tensor)

    def calculate_td_estimate(self, states, actions):
        actions = tf.cast(actions, dtype=tf.int32)
        current_online_Q = self(states, network=self.ONLINE)
        return tf.gather(current_online_Q, actions, batch_dims=1)

    def calculate_td_target(self, next_states, rewards, dones):
        dones = tf.cast(dones, dtype=tf.float32)
        next_online_Q = self(next_states, network=self.ONLINE)
        best_actions = tf.math.argmax(next_online_Q, axis=1)
        
        next_target_Q = self(next_states, network=self.TARGET)

        return rewards + self.gamma * tf.math.multiply(tf.gather(next_target_Q, best_actions, batch_dims=1), (1 - dones))

    @tf.function
    def train_step(self, data):
        states, actions, next_states, rewards, dones = data[0]

        with tf.GradientTape() as tape:
            # Calculate the TD estimate and TD target
            td_estimate = self.calculate_td_estimate(states, actions)
            td_target = self.calculate_td_target(next_states, rewards, dones)

            # Calculate loss
            loss = self.compiled_loss(td_target, td_estimate, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        self.compiled_metrics.update_state(td_target, td_estimate)

        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        return {"observation_shape": self.observation_shape, "num_actions": self.num_actions, "gamma": self.gamma}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def sync_target(self):
        self.target.set_weights(self.online.get_weights())