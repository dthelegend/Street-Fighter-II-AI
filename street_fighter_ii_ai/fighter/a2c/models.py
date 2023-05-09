import tensorflow as tf

class ActorCritic(tf.keras.Model):
    
    def __init__(self, observation_shape: tuple, num_actions: int):
        super().__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        self.net = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=observation_shape),
            tf.keras.layers.Conv2D(32, 8, 4, activation="relu"),
            tf.keras.layers.Conv2D(64, 4, 2, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        
        self.actor = tf.keras.layers.Dense(num_actions, activation='softmax')
        self.critic = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        x = self.net(inputs)
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output
    
    @tf.function
    def train_step(self, data):
        return super().train_step(data)