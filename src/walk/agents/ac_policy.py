from .abstract_env import AbstractHumanoidEnv
from ..models.actor_critic import ActorCritic
from ..utils.memory import Memory
import numpy as np

class AC_Policy(AbstractHumanoidEnv):

    def __init__(self, args):
        super(AC_Policy, self).__init__(args, "Actor Critic algorithm")
        self.memory = Memory()

        self.actor_model = ActorCritic.actor_model(self.env)
        self.target_actor_model = ActorCritic.actor_model(self.env)

        self.critic_model = ActorCritic.critic_model(self.env)
        self.target_critic_model = ActorCritic.critic_model(self.env)

    def train():
        pass

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
