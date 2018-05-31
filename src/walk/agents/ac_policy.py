from .abstract_env import AbstractHumanoidEnv
from ..models.actor_critic import Actor, Critic
from ..utils.memory import Memory
from ..utils.noise import Noise
import numpy as np
import tensorflow as tf

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
        self.tf_session = tf.Session()
        self.noise = Noise(mu=np.zeros(self.env.action_space.shape[0]))
        self.actor_file = "actor.ckpt"
        self.critic_file = "critic.ckpt"
        self.ratio_random_action = [0, 0]
        self.current_epsilon = self.params.epsilon

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
                                     self.params.actor_batch_norm)
            self.actor_loss = 0

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
                                       self.params.critic_batch_norm)
            self.critic_loss = 0

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
                                            self.params.actor_batch_norm)
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
                                              self.params.critic_batch_norm)

        self.target_params = tf.trainable_variables(scope="target_network")
        self.net_params = tf.trainable_variables(scope="network")
        print("target params:", len(self.target_params))
        print("network params:", len(self.net_params))

        self.update_target_net_params = \
            [self.target_params[i].assign(tf.multiply(self.net_params[i],
                                                      self.params.tau) +
                                          tf.multiply(self.target_params[i],
                                                      1. - self.params.tau))
             for i in range(len(self.target_params))]
        self.just_copy_target_net_params = \
            [self.target_params[i].assign(self.net_params[i])
             for i in range(len(self.target_params))]

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

        decay = self.params.epsilon_decay

        # Decaying function
        self.current_epsilon = self.current_epsilon * (
            (1 - decay) ** (self.t % self.params.epochs))

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

            if self.params.action_range:
                next_actions = (next_actions +
                                self.params.action_range/2
                ) / self.params.action_range

            # Compute the Q+1 value with next s+1 and a+1
            Q_next = self.tf_session.run(self.target_critic_model.Q, feed_dict={
                self.target_critic_model.input_state_ph: next_states,
                self.target_critic_model.input_action_ph: next_actions
            })

            # gamma is the discounted factor
            Q_next = self.params.gamma * Q_next * (1 - dones)
            Q_next = Q_next + rewards

            # Train the critic network and get gradients
            feed_critic = {
                self.critic_model.input_state_ph: states,
                self.critic_model.input_action_ph: actions,
                self.critic_model.true_target_ph: Q_next
            }
            self.critic_loss, _, critic_action_gradient = self.tf_session.run(
                [self.critic_model.loss, self.critic_model.opt,
                 self.critic_model.action_gradients],
                feed_dict=feed_critic)

        with tf.variable_scope("train_actor"):
            # Train the actor network with the critic gradients
            feed_actor = {
                self.actor_model.input_ph: states,
                self.actor_model.action_gradients: critic_action_gradient[0]
            }
            self.actor_loss, _ = self.tf_session.run([self.actor_model.loss,
                                                      self.actor_model.opt],
                                                     feed_dict=feed_actor)

        with tf.variable_scope("soft_update"):
            # Update target network
            self._update_target_network()
        # Save weights of the models
        self.actor_model.save_model_weights(self.tf_session, "actor.ckpt")
        self.critic_model.save_model_weights(self.tf_session, "critic.ckpt")

    def _update_target_network(self, just_copy=False):
        """Update target network with soft update if just_copy is False
        and just copy weights from model to target if True.
        """
        if just_copy:
            self.tf_session.run(self.just_copy_target_net_params)
        else:
            self.tf_session.run(self.update_target_net_params)

    def act(self, state):
        """Return action given a state.
        Implements epsilon-greedy exploration.
        """
        if self.params.train:
            # Return a random action epsilon percent of the time
            if np.random.random() < self.params.epsilon:
                # +1 to random actions taken
                self.ratio_random_action[0] = self.ratio_random_action[0] + 1
                return self.env.action_space.sample()

        # +1 to action which comes out from network
        self.ratio_random_action[1] = self.ratio_random_action[1] + 1

        # Predict action from state
        reshaped_state = (state.reshape(1,
                                        self.env.observation_space.shape[0])
                          + self.params.state_range/2) / self.params.state_range
        feed = {self.actor_model.input_ph: reshaped_state}
        return self.tf_session.run(self.actor_model.output, feed)[0]

    def run(self):
        """Run the simulation.
        """
        print(self.params)
        state = self.env.reset()

        # True to initialize actor and critic with saved weights
        if self.params.load_weights:
            self.actor_model.load_model_weights(
                self.tf_session, self.actor_file)
            self.critic_model.load_model_weights(
                self.tf_session, self.critic_file)

        for j in range(self.params.epochs):
            action = self.act(state)
            if self.params.noisy:
                action += self.noise()

            print("Number of random taken actions",
                  self.ratio_random_action[0],
                  "against", self.ratio_random_action[1],
                  "(Total:", sum(self.ratio_random_action), ")")

            new_state, reward, done, info = self.env.step(action)

            # Put the current environment in the memory
            # State interval is [-5;5] and action range is [-1;1]
            self.memory.remember(state, action, reward * self.params.reward_multiply,
                                 new_state, done, state_range=10, action_range=2)

            # Train the network
            self.train()

            # Reset the environment if done
            self.reset(done)

            # Render the environment
            self.render()

            # Plot needed values
            self.plotting(state=state, reward=reward,
                          epsilon=self.current_epsilon,
                          c_loss=self.critic_loss,
                          a_loss=self.actor_loss)

            # Change current state
            state = new_state

        # Save the plot
        self.board.save()
