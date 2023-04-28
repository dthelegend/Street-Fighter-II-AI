import tensorflow as tf
import numpy as np

class DuelingDeepQNetwork(tf.keras.Model):
    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.net = tf.keras.models.Sequential([
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
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        
        self.online = DuelingDeepQNetwork(num_actions)
        self.target = DuelingDeepQNetwork(num_actions)
        
        self.gamma = 0.9

    @tf.function
    def call(self, input_tensor):
        return self.online(input_tensor)

    def calculate_td_estimate(self, states, actions):
            current_online_Q = self.online(states)
            return current_online_Q[:,actions]

    def calculate_td_target(self, next_states, rewards, dones):
        next_online_Q = self.online(next_states)
        best_actions = tf.math.argmax(next_online_Q, axis=1)
        
        next_target_Q = self.target(next_states)

        return rewards + self.gamma * tf.math.multiply(next_target_Q[:,best_actions], (1 - tf.cast(dones, tf.float32)))

    @tf.function
    def train_step(self, data):

        states, actions, next_states, rewards, dones = data[0]
        actions = tf.get_static_value(actions)

        with tf.GradientTape() as tape:
            # Calculate the TD estimate and TD target
            td_estimate = self.calculate_td_estimate(states, actions)
            td_target = self.calculate_td_target(next_states, rewards, dones)

            # Calculate loss
            loss = self.compute_loss(td_target, td_estimate, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.online.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        self.compiled_metrics.update_state(td_target, td_estimate)

        return {m.name: m.result() for m in self.metrics}

    def sync(self):
        self.target.set_weights(self.online.get_weights())