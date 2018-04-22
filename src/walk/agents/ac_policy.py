from .abstract_env import AbstractHumanoidEnv
from ..models.actor_critic import ActorCritic
from ..utils.memory import Memory
import numpy as np
import keras.backend as K
import tensorflow as tf

# https://arxiv.org/pdf/1607.07086.pdf

class AC_Policy(AbstractHumanoidEnv):

    def __init__(self, args):
        super(AC_Policy, self).__init__(args, "Actor Critic algorithm")
        self.memory = Memory()
        self.tf_session = tf.Session()
        K.set_session(self.tf_session)
        self.actor_file = "actor.h5"
        self.critic_file = "critic.h5"
        self.model_builder = ActorCritic(self.env.observation_space,
                                         self.env.action_space)

        ################################################################
        # ACTOR
        ################################################################
        self.actor_input, self.actor_model = self.model_builder.actor_model()
        _, self.target_actor_model = self.model_builder.actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[0]])

        actor_model_weights = self.actor_model.trainable_weights

        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        self.actor_optimizer = tf.train.AdamOptimizer(
            self.params.learning_rate).apply_gradients(grads)

        ################################################################
        # CRITIC
        ################################################################
        self.critic_state_input, self.critic_action_input,\
            self.critic_model = self.model_builder.critic_model()

        _, _, self.target_critic_model = self.model_builder.critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)
        self.tf_session.run(tf.global_variables_initializer())

    def _train_actor(self, samples):
        for sample in samples:
            state, action, reward, new_state, done = sample
            reshaped_state = state.reshape(1, self.env.observation_space.shape[0])

            next_action = self.actor_model.predict(reshaped_state)
            reshaped_next_action = action.reshape(1, self.env.action_space.shape[0])

            grads = self.tf_session.run(self.critic_grads,
                                        feed_dict = {
                                            self.critic_state_input: reshaped_state,
                                            self.critic_action_input: reshaped_next_action
                                        })[0]

            self.tf_session.run(self.actor_optimizer,
                                feed_dict = {
                                    self.actor_input: reshaped_state,
                                    self.actor_critic_grad: grads
                                })
        self.model_builder.save_model_weights(self.actor_model, self.actor_file)


    def _train_critic(self, samples):
        for sample in samples:
            state, action, reward, new_state, done = sample
            reshaped_state = state.reshape(1, self.env.observation_space.shape[0])
            reshaped_new_state = new_state.reshape(1, self.env.observation_space.shape[0])

            # Get the action the actor will take with new state
            reshaped_action = action.reshape(1, self.env.action_space.shape[0])
            next_action = self.target_actor_model.predict(reshaped_new_state)
            reshaped_next_action = action.reshape(1, self.env.action_space.shape[0])

            # Get the Q_+1-value
            Q_value_p1 = self.target_critic_model.predict(
                [reshaped_new_state, reshaped_next_action])[0][0]
            # Don't see in the future on action which lead to done
            future_rewards = reward + self.params.gamma * Q_value_p1 * (1 - done)
            self.critic_model.fit([reshaped_state, reshaped_action], [future_rewards],
                                  verbose=0,
                                  callbacks=self.model_builder.callbacks(self.critic_file))

    def _update_target_network(self, model, target):
        weights = model.get_weights()
        target_weights = target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i] * self.params.tau \
                                + (1 - self.params.tau) * weights[i]

        target.set_weights(target_weights)

    def train(self):

        if not self.params.train:
            return

        # Don't train if there is not enough samples in the memory
        if len(self.memory) < self.params.batch_size:
            return

        samples = self.memory.samples(self.params.batch_size)

        # Let's try first to run samples on critic then on actor
        # And after one sample per sample
        self._train_critic(samples)
        self._train_actor(samples)

        self._update_target_network(self.actor_model, self.target_actor_model)
        self._update_target_network(self.critic_model, self.target_critic_model)

    def act(self, state):
        epsilon = self.params.epsilon
        decay = self.params.epsilon_decay
        self.params.epsilon = epsilon * ((1 - decay) ** self.t)
        if np.random.random() < self.params.epsilon:
            return self.env.action_space.sample()
        reshaped_state = state.reshape(1, self.env.observation_space.shape[0])
        return self.actor_model.predict(reshaped_state)[0]

    def run(self, train_pass=10000, epochs=100):
        state = self.env.reset()
        
        if self.params.load_weights:
            self.model_builder.load_model_weights(self.actor_model, self.actor_file)
            self.model_builder.load_model_weights(self.critic_model, self.critic_file)

        for i in range(train_pass):
            if self.params.reset:
                self.env.reset()

            for j in range(epochs):
                action = self.act(state)
                new_state, reward, done, info = self.env.step(action)

                # Should be reshaped for training after sending it to the environment
                #action = action.reshape(1, self.env.action_space.shape[0])
                #new_state = new_state.reshape(1, self.env.observation_space.shape[0])

                self.memory.remember(state, action, reward, new_state, done)

                self.reset(done)

                self.render()

                self.plotting(state, self.params.epsilon, reward)

                state = new_state

                self.train()
