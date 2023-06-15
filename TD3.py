import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        ''' inicjalizacja buffera '''
        self.buffer_size = buffer_size # rozmiar buffera
        self.counter     = 0 # licznik

        # zainicjalizowanie macierzami zerowymi 
        self.state_mem      = np.zeros((self.buffer_size, *input_shape))
        self.new_state_mem  = np.zeros((self.buffer_size, *input_shape))
        self.action_mem     = np.zeros((self.buffer_size, n_actions))
        self.reward_mem     = np.zeros(self.buffer_size)
        self.terminal_mem   = np.zeros(self.buffer_size, dtype=np.bool)

    def add(self, state, action, reward, next_state, done):
        ''' operacja przejścia '''
        idx = self.counter % self.buffer_size # przepełnienie buffera

        # zapis wartości z epizodu do buffera
        self.state_mem[idx]     = state
        self.action_mem[idx]    = action
        self.reward_mem[idx]    = reward
        self.new_state_mem[idx] = next_state
        self.terminal_mem[idx]  = done

        self.counter += 1

    def sample(self, batch_size):
        ''' losowa próbka z batch '''
        buffer_max_memory = min(self.counter, self.buffer_size)

        batch = np.random.choice(buffer_max_memory, batch_size)

        states      = self.state_mem[batch]
        next_states = self.new_state_mem[batch]
        actions     = self.action_mem[batch]
        rewards     = self.reward_mem[batch]
        dones       = self.terminal_mem[batch]

        return states, actions, rewards, next_states, dones
    
class ActorNetwork(tf.keras.Model):
    ''' Aktor ocenia aktualny stan i zwraca prawdopodieństwo poprawności stanów '''
    def __init__(self, max_action, name):
        super(ActorNetwork, self).__init__() # inicjalizacja modelu keras
        
        # rozmiary sieci 400, 300 z publikacji TD3
        self.fc1 = tf.keras.layers.Dense(400, activation='relu')
        self.fc2 = tf.keras.layers.Dense(300, activation='relu')
        self.fc3 = tf.keras.layers.Dense(max_action, activation='tanh')
        
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.model_name = name
        self.max_action = max_action

    def call(self, state):
        ''' po przejściu przez sieci fc1, fc2, fc3 
        otrzymujemy deterministyczne akcje na podstawie środowiska'''
        x = self.fc1(state)
        x = self.fc2(x)
        action = self.fc3(x)

        return action


class CriticNetwork(tf.keras.Model):
    def __init__(self, name):
        super(CriticNetwork, self).__init__()  # inicjalizacja modelu keras
        self.fc1 = tf.keras.layers.Dense(400, activation='relu')
        self.fc2 = tf.keras.layers.Dense(300, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)
        self.model_name =  name

        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)


    def call(self, state, action):
        x = tf.concat([state, action], axis=1)

        q1 = self.fc1(x)
        q1 = self.fc2(q1)
        q = self.q(q1)

        return q


class TD3Agent:
    def __init__(self, input_dims, env, batch_size, n_actions=2):
        self.gamma  = 0.99
        self.tau    = 0.005 # wg dokumentacji tau << 1
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.memory         = ReplayBuffer(1000000, input_dims, n_actions)
        self.batch_size     = batch_size
        self.learn_counter  = 0
        self.time_step      = 0
        self.warmup         = 1000
        self.n_actions      = n_actions
        self.actor_interval = 2

        self.actor          = ActorNetwork(max_action=n_actions, name='actor')

        self.critic_1       = CriticNetwork(name='critic_1')
        self.critic_2       = CriticNetwork(name='critic_2')

        self.target_actor   = ActorNetwork(max_action=n_actions, name='target_actor')

        self.target_critic_1 = CriticNetwork(name='target_critic_1')
        self.target_critic_2 = CriticNetwork(name='target_critic_2')

        # kompilowanie modeli
        self.actor.compile(optimizer=Adam(learning_rate=0.001), loss='mean')
        
        self.critic_1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=0.001), loss='mean')

        self.target_critic_1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        self.noise = 0.1
        self.update(tau = 1)
    
    def choose_action(self, observation):
        # warmup
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            # konwersja do wektora
            mu = self.actor(state)[0]
        mu_ = mu + np.random.normal(scale=self.noise)
        mu_ = tf.clip_by_value(mu_, self.min_action, self.max_action)
        self.time_step += 1

        return mu_
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.add(state, action, reward, new_state, done)
    
    def train(self):
        if self.memory.counter < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as grad:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + \
    tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)
        
            q1_ = self.target_critic_1(states_, target_actions)
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q1_ = tf.squeeze(q1_, 1)

            q2_ = self.target_critic_2(states_, target_actions)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_1_loss = tf.keras.losses.MSE(target, q1)
            critic_2_loss = tf.keras.losses.MSE(target, q2)
        

        critic_1_gradient = grad.gradient(critic_1_loss,
                                          self.critic_1.trainable_variables)
        critic_2_gradient = grad.gradient(critic_2_loss,
                                          self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(
                   zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
                   zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_counter += 1

        if self.learn_counter % self.actor_interval != 0:
            return

        with tf.GradientTape() as grad:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = grad.gradient(actor_loss,
                                       self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
                        zip(actor_gradient, self.actor.trainable_variables))

        self.update()

        

    def update(self, tau = None):
        if tau is None:
            tau = self.tau
        
        network_weights = self.actor.weights
        target_weights  = self.target_actor.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_actor.set_weights(new_weights)

        network_weights = self.critic_1.weights
        target_weights  = self.target_critic_1.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_critic_1.set_weights(new_weights)

        network_weights = self.critic_2.weights
        target_weights  = self.target_critic_2.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_critic_2.set_weights(new_weights)



    # def get_action(self, state):
    #     state = np.expand_dims(state, axis=0)
    #     action = self.actor(state)
    #     action = tf.squeeze(action, axis=0)
    #     return action.numpy()

    def save_models(self, checkpoint_dir):
        print('SAVING')
        self.actor.save_weights(os.path.join(checkpoint_dir, "actor_episode.ckpt"))
        self.critic_1.save_weights(os.path.join(checkpoint_dir, "critic_1_episode.ckpt"))
        self.critic_2.save_weights(os.path.join(checkpoint_dir, "critic_2_episode.ckpt"))
        self.target_actor.save_weights(os.path.join(checkpoint_dir, "target_actor_episode.ckpt"))
        self.target_critic_1.save_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))
        self.target_critic_2.save_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))

    def load_models(self, checkpoint_dir):
        print('LOADING')
        self.actor.load_weights(os.path.join(checkpoint_dir, "actor_episode.ckpt"))
        self.critic_1.load_weights(os.path.join(checkpoint_dir, "critic_1_episode.ckpt"))
        self.critic_2.load_weights(os.path.join(checkpoint_dir, "critic_2_episode.ckpt"))
        self.target_actor.load_weights(os.path.join(checkpoint_dir, "target_actor_episode.ckpt"))
        self.target_critic_1.load_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))
        self.target_critic_2.load_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))
