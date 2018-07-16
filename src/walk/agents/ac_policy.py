""" Actor critic agent definition.
"""
import os
import tensorflow as tf
import numpy as np
from ..utils.memory import Memory
from ..models.actor_critic import Actor, Critic
from ..utils.array_utils import fit_normalize
from .abstract_env import AbstractHumanoidEnv, AbstractMountainCarEnv
from ..utils.noise import Noise

# https://arxiv.org/pdf/1607.07086.pdf


class AC_Policy(AbstractHumanoidEnv):
    """Actor critic agent. Implements DDPG algorithm from
    https://arxiv.org/pdf/1509.02971v5.pdf.
    """

    def __init__(self, args):
        """Initialize all the needed components and build
        the network.
        """
        super(AC_Policy, self).__init__(args)
        self.memory = Memory()
        if self.params.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.tf_session = tf.Session()

        self.noise = Noise(mu=np.zeros(self.env.action_space.shape[0]))
        self.actor_file = "actor"
        self.critic_file = "critic"
        self.ratio_random_action = [0, 0]
        self.scale = np.vectorize(fit_normalize)

        with tf.variable_scope("network"):
            ################################################################
            # ACTOR
            ################################################################
            self.actor_model = Actor(self.env.observation_space,
                                     self.env.action_space,
                                     self.params.actor_layers,
                                     self.params.actor_learning_rate,
                                     self.params.tau,
                                     self.params.batch_size,
                                     "network/actor",
                                     self.params.dropout,
                                     self.params.actor_batch_norm,
                                     sub_folder=self.name_run)
            self.actor_loss = 0
            self.a_params = tf.trainable_variables()

            ################################################################
            # CRITIC
            ################################################################
            self.critic_model = Critic(self.env.observation_space,
                                       self.env.action_space,
                                       self.params.critic_layers,
                                       self.params.critic_learning_rate,
                                       self.params.tau,
                                       self.params.batch_size,
                                       "network/critic",
                                       self.params.dropout,
                                       self.params.critic_batch_norm,
                                       sub_folder=self.name_run)
            self.critic_loss = 0
            self.c_params = tf.trainable_variables()[len(self.a_params):]

        ################################################################
        # TARGET NETWORK
        ################################################################

        with tf.variable_scope("target_network"):
            # Get a second actor model to use it for target and copy the
            # weight of the first actor model to it
            self.target_actor_model = Actor(self.env.observation_space,
                                            self.env.action_space,
                                            self.params.actor_layers,
                                            self.params.actor_learning_rate,
                                            self.params.tau,
                                            self.params.batch_size,
                                            "target_network/actor",
                                            self.params.dropout,
                                            self.params.actor_batch_norm,
                                            trainable=False)
            self.at_params = tf.trainable_variables()[
                len(self.c_params) + len(self.a_params):]
            # Get a second critic model to use it for target and copy their
            # weight of the first critic model to it
            self.target_critic_model = Critic(self.env.observation_space,
                                              self.env.action_space,
                                              self.params.critic_layers,
                                              self.params.critic_learning_rate,
                                              self.params.tau,
                                              self.params.batch_size,
                                              "target_network/critic",
                                              self.params.dropout,
                                              self.params.critic_batch_norm,
                                              trainable=False)
            self.ct_params = tf.trainable_variables()[
                len(self.at_params) + len(self.c_params) + len(self.a_params):]

        print("************** ACTOR PARAMS",
              len(self.a_params),
              "**********************")
        print(self.a_params)
        print("************** CRITIC PARAMS",
              len(self.c_params),
              "*********************")
        print(self.c_params)
        print("********** TARGET ACTOR PARAMS",
              len(self.at_params),
              "******************")
        print(self.at_params)
        print("********** TARGET CRITIC PARAMS",
              len(self.ct_params),
              "*****************")
        print(self.ct_params)

        # Defines the soft target update operations
        self.update_critic_target = \
            [self.ct_params[i].assign(tf.multiply(self.c_params[i],
                                                  self.params.tau) +
                                      tf.multiply(self.ct_params[i],
                                                  1. - self.params.tau))
             for i in range(len(self.ct_params))]
        self.update_actor_target = \
            [self.at_params[i].assign(tf.multiply(self.a_params[i],
                                                  self.params.tau) +
                                      tf.multiply(self.at_params[i],
                                                  1. - self.params.tau))
             for i in range(len(self.at_params))]

        # Defines the initialization of the target parameter operations
        self.init_critic_target = \
            [self.ct_params[i].assign(self.c_params[i])
             for i in range(len(self.ct_params))]
        self.init_actor_target = \
            [self.at_params[i].assign(self.a_params[i])
             for i in range(len(self.at_params))]

        # Defines the plotting library
        if self.params.plot == "tensorflow":
            self.use_tensorboard(self.tf_session)
        if self.params.plot == "matplotlib":
            self.use_matplotlib("Actor Critic algorithm")

        # Initialize global variables of the session
        self.tf_session.run(tf.global_variables_initializer())
        self._update_target_network(just_copy=True)

    def train(self):
        """Train both network if asked to when the memory
        is full enough.
        Train and update the target network with soft update.
        """
        if not self.params.train:
            return

        # Don't train if there is not enough samples in the memory
        if len(self.memory) < self.params.batch_size:
            return

        # Get samples of memory
        states, actions, rewards, next_states, dones = \
            self.memory.samples(self.params.batch_size)

        with tf.variable_scope("train_critic"):

            # Predicted actions
            next_actions = self.tf_session.run(
                self.target_actor_model.output,
                feed_dict={
                    self.target_actor_model.input_ph: next_states
                })
            next_actions = self.scale(next_actions, self.act_low, self.act_high)

            # Compute the Q+1 value with next s+1 and a+1
            Q_next = self.tf_session.run(
                self.target_critic_model.Q,
                feed_dict={
                    self.target_critic_model.input_state_ph: next_states,
                    self.target_critic_model.input_action_ph: next_actions
                })

            # gamma is the discounted factor
            Q_next = self.params.gamma * Q_next * (1 - dones)
            Q_next = np.add(Q_next, rewards)

            # Train the critic network and get gradients
            feed_critic = {
                self.critic_model.input_state_ph: states,
                self.critic_model.input_action_ph: actions,
                self.critic_model.true_target_ph: Q_next
            }
            self.critic_loss, _, critic_action_gradient = \
                self.tf_session.run(
                    [self.critic_model.loss, self.critic_model.opt,
                     self.critic_model.action_gradients],
                    feed_dict=feed_critic)

        with tf.variable_scope("train_actor"):
            # Train the actor network with the critic gradients
            feed_actor = {
                self.actor_model.input_ph: states,
                self.actor_model.action_gradients: critic_action_gradient[0]
            }
            _ = self.tf_session.run([self.actor_model.opt],
                                    feed_dict=feed_actor)
            # self.actor_loss, _ = self.tf_session.run([self.actor_model.loss,
            #                                           self.actor_model.opt],
            #                                          feed_dict=feed_actor)

        with tf.variable_scope("soft_update"):
            # Update target network
            self._update_target_network()
        # Save weights of the models
        self.actor_model.save_model_weights(self.tf_session, self.actor_file)
        self.critic_model.save_model_weights(self.tf_session, self.critic_file)

    def _update_target_network(self, just_copy=False):
        """Update target network with soft update if just_copy is False
        and just copy weights from model to target if True.
        """
        if just_copy:
            self.tf_session.run(self.init_actor_target)
            self.tf_session.run(self.init_critic_target)
        else:
            self.tf_session.run(self.update_actor_target)
            self.tf_session.run(self.update_critic_target)

    def act(self, state):
        """Return action given a state.
        Implements epsilon-greedy exploration.
        """
        # Predict action from state
        reshaped_state = state.reshape(1, self.env.observation_space.shape[0])
        feed = {self.actor_model.input_ph: reshaped_state}
        actions = self.tf_session.run(self.actor_model.output, feed)[0]
        return self.scale(actions, self.act_low, self.act_high)

    def reset(self):
        super(AC_Policy, self).reset()

    def run(self):
        """Run the simulation.
        """
        print(self.params)

        # True to initialize actor and critic with saved weights
        if self.params.load_weights is not None:
            self.actor_model.load_model_weights(
                self.tf_session, self.params.load_weights, self.actor_file)
            self.critic_model.load_model_weights(
                self.tf_session, self.params.load_weights, self.critic_file)

        seed = 42
        np.random.seed(seed)
        tf.set_random_seed(seed)

        for j in range(self.params.epochs):
            # Reset the environment if done
            self.reset()
            state = self.env.reset()

            for i in range(self.params.pass_per_epoch):
                print("EPOCHS:", j, "PASS:", i)
                action = self.act(state)
                if self.params.noisy and j < self.params.noise_threshold:
                    action += self.noise()

                new_state, reward, done, _ = self.env.step(action)

                # Put the current environment in the memory
                # State interval is [-5;5] and action range is [-1;1]
                self.memory.remember(state, action,
                                     reward * self.params.reward_multiply,
                                     new_state, done)

                # Train the network
                self.train()

                # Render the environment
                self.render()

                # Plot needed values
                self.plotting(state=state, reward=reward,
                              c_loss=self.critic_loss,
                              a_loss=self.actor_loss,
                              epoch=j)

                if done:
                    break;

                # Change current state
                state = new_state

        # Save the plot
        self.board.save()
