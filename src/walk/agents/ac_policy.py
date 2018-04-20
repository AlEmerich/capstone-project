from .abstract_env import AbstractHumanoidEnv
from ..models.actor_critic import ActorCritic
from ..utils.memory import Memory
import numpy as np
import tensorflow as tf

class AC_Policy(AbstractHumanoidEnv):

    def __init__(self, args):
        super(AC_Policy, self).__init__(args, "Actor Critic algorithm")
        self.memory = Memory()

        self.actor_model = ActorCritic.actor_model(self.env)
        self.target_actor_model = ActorCritic.actor_model(self.env)

        self.critic_model = ActorCritic.critic_model(self.env)
        self.target_critic_model = ActorCritic.critic_model(self.env)

    def _train_actor(self, state, action, reward, new_state, done):
        for sample in samples:
            state, action, reward, new_state, done = sample

            

    def _train_critic(self, samples):
        for sample in samples:
            state, action, reward, new_state, done = sample

            # Get the Q-value of the action given the state
            Q_value = self.critic_model.predict([state, action])

            # Get the action the actor will take with new state
            next_action = self.target_actor_model(new_state)

            # Get the Q_+1-value
            Q_value_p1 = self.target_critic_model([new_state, next_action])

            # Don't see in the future on action which lead to done
            future_rewards = reward + self.params.gamma * Q_value_p1 * (1 - done)

            self.critic_model.fit([state, action], future_rewards)


    def train(self):

        # Don't train if there is not enough samples in the memory
        if len(self.memory) < self.batch_size:
            return

        samples = self.memory.samples(self.batch_size)

        # Let's try first to run samples on critic then on actor
        # And after one sample per sample
        self._train_critic(samples)
        self._train_actor(samples)

    def act(self, state):
        epsilon = self.params.epsilon
        decay = self.params.epsilon_decay
        self.params.epsilon = epsilon * ((1 - decay) ** self.t)
        if np.random.random() < self.params.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(state)

    def run(self, train_pass=10000, epochs=100):
        state = self.env.reset()

        for i in range(train_pass):
            if self.params.reset:
                self.env.reset()

            for j in range(epochs):
                action = self.act(state)

                new_state, reward, done, info = self.env.step(action)

                self.memory.remember(state, action, reward, new_state, done)

                state = new_state

                self.train()
