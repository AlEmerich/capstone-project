""" Actor critic agent definition.
https://arxiv.org/pdf/1607.07086.pdf
"""
import os
import tensorflow as tf
import numpy as np
import math
from ..utils.array_utils import scale
from ..utils.memory import Memory
from ..models.actor_critic import Actor, Critic
from ..utils.noise import Noise
# from tensorflow.python.client import timeline


def environmentFactory(abstract_env):
    class AC_Policy(abstract_env):
        """Actor critic agent. Implements DDPG algorithm from
        https://arxiv.org/pdf/1509.02971v5.pdf.
        """

        def __init__(self, args, name_run):
            """Initialize all the needed components and build
            the network.
            """
            super(AC_Policy, self).__init__(args, name_run)

            self._save_params_info(self.saved_folder)

            self.memory = Memory()

            self.__init_session__()

            self.noise = Noise(mu=np.zeros(self.env.action_space.shape[0]),
                               sigma=self.params.sigma,
                               theta=self.params.theta)
            self.actor_file = "actor"
            self.critic_file = "critic"

            self.decaying_noise = True
            if not self.params.initial_noise_scale or \
               not self.params.noise_decay:
                self.decaying_noise = False

            self.__init_networks__()

            self.actor_saver = tf.train.Saver(
                var_list=self.actor_model.network_params)
            self.critic_saver = tf.train.Saver(
                var_list=self.critic_model.network_params)

            self.__init_soft_target_ops__()

            # Defines the plotting library
            if self.params.plot == "tensorflow":
                self.use_tensorboard(self.tf_session)
            if self.params.plot == "matplotlib":
                self.use_matplotlib("Actor Critic algorithm")

            # Initialize global variables of the session
            self.tf_session.run(tf.global_variables_initializer())
            self._update_target_network(just_copy=True)

        def __init_session__(self):
            """Instantiate Tensorflow session.
            """
            if self.params.device == "cpu":
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            gpu_options = tf.GPUOptions(allow_growth=True)
            self.tf_session = tf.Session(config=tf.ConfigProto(
                inter_op_parallelism_threads=4,
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=gpu_options))
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

        def __init_networks__(self):
            """Instantiate Neural net, actor and critic, with targets.
            """
            with tf.variable_scope("network"):
                #####################################################
                # ACTOR
                #####################################################
                self.actor_model = Actor(
                    self.env.observation_space,
                    self.env.action_space,
                    self.params.actor_layers,
                    self.params.actor_learning_rate,
                    self.params.tau,
                    self.params.batch_size,
                    "network/actor",
                    self.params.dropout,
                    self.params.actor_batch_norm,
                    self.saved_folder)
                self.actor_loss = 0

                #####################################################
                # CRITIC
                #####################################################
                self.critic_model = Critic(
                    self.env.observation_space,
                    self.env.action_space,
                    self.params.critic_layers,
                    self.params.critic_learning_rate,
                    self.params.tau,
                    self.params.batch_size,
                    "network/critic",
                    self.params.dropout,
                    self.params.critic_batch_norm,
                    self.saved_folder)
                self.critic_loss = 0

            #####################################################
            # TARGET NETWORK
            #####################################################
            with tf.variable_scope("target_network"):
                # Get a second actor model to use it for target and copy the
                # weight of the first actor model to it
                self.target_actor_model = Actor(
                    self.env.observation_space,
                    self.env.action_space,
                    self.params.actor_layers,
                    self.params.actor_learning_rate,
                    self.params.tau,
                    self.params.batch_size,
                    "target_network/actor",
                    self.params.dropout,
                    self.params.actor_batch_norm,
                    self.saved_folder,
                    trainable=False)

                # Get a second critic model to use it for target and copy their
                # weight of the first critic model to it
                self.target_critic_model = Critic(
                    self.env.observation_space,
                    self.env.action_space,
                    self.params.critic_layers,
                    self.params.critic_learning_rate,
                    self.params.tau,
                    self.params.batch_size,
                    "target_network/critic",
                    self.params.dropout,
                    self.params.critic_batch_norm,
                    self.saved_folder,
                    trainable=False)

        def __init_soft_target_ops__(self):
            """Defines soft and hard target updates operations.
            """
            # Defines the soft target update operations
            self.update_critic_target = [
                self.target_critic_model.network_params[i].assign(
                    tf.multiply(self.critic_model.network_params[i],
                                self.params.tau) +
                    tf.multiply(self.target_critic_model.network_params[i],
                                1. - self.params.tau))
                for i in range(len(self.target_critic_model.network_params))]
            self.update_actor_target = [
                self.target_actor_model.network_params[i].assign(
                    tf.multiply(self.actor_model.network_params[i],
                                self.params.tau) +
                    tf.multiply(self.target_actor_model.network_params[i],
                                1. - self.params.tau))
                for i in range(len(self.target_actor_model.network_params))]

            # Defines the initialization of the target parameter operations
            self.init_critic_target = [
                self.target_critic_model.network_params[i].assign(
                    self.critic_model.network_params[i])
                for i in range(len(self.target_critic_model.network_params))]
            self.init_actor_target = [
                self.target_actor_model.network_params[i].assign(
                    self.actor_model.network_params[i])
                for i in range(len(self.target_actor_model.network_params))]

        def __print_params__(self):
            """Print parameters of neural nets.
            """
            print("************** ACTOR PARAMS",
                  len(self.actor_model.network_params),
                  "**********************")
            print(self.actor_model.network_params)
            print("************** CRITIC PARAMS",
                  len(self.critic_model.network_params),
                  "*********************")
            print(self.critic_model.network_params)
            print("********** TARGET ACTOR PARAMS",
                  len(self.target_actor_model.network_params),
                  "******************")
            print(self.target_actor_model.network_params)
            print("********** TARGET CRITIC PARAMS",
                  len(self.target_critic_model.network_params),
                  "*****************")
            print(self.target_critic_model.network_params)

        def tf_run_op(self, op, feed_dict=None):
            return self.tf_session.run(op,
                                       feed_dict=feed_dict,
                                       options=self.options,
                                       run_metadata=self.run_metadata)

        def train(self):
            """Train both network if asked to when the memory
            is full enough.
            Train and update the target network with soft update.
            """
            if not self.params.train:
                return

            # Get samples of memory
            exp = self.memory.samples(self.params.batch_size)
            states, actions, rewards, next_states, dones = exp

            with tf.variable_scope("train_critic"):

                # Predicted actions
                next_actions = self.tf_run_op(
                    self.target_actor_model.output,
                    feed_dict={
                        self.target_actor_model.input_ph: next_states
                    })
                next_actions = scale(next_actions, self.act_low, self.act_high)

                # Compute the Q+1 value with next s+1 and a+1
                Q_next = self.tf_run_op(
                    self.target_critic_model.Q,
                    feed_dict={
                        self.target_critic_model.input_state_ph: next_states,
                        self.target_critic_model.input_action_ph: next_actions
                    })

                # gamma is the discounted factor
                expected = rewards + self.params.gamma * Q_next * (1-dones)

                # Train the critic network and get gradients
                self.critic_loss, _ = self.tf_run_op(
                    [self.critic_model.loss, self.critic_model.opt],
                    feed_dict={
                        self.critic_model.input_state_ph: states,
                        self.critic_model.input_action_ph: actions,
                        self.critic_model.true_target_ph: expected
                    })

                act_for_grad = scale(self.tf_run_op(
                    self.actor_model.output,
                    feed_dict={
                        self.actor_model.input_ph: states
                    }), self.act_low, self.act_high)

                action_gradient = self.tf_run_op(
                    self.critic_model.action_gradients,
                    feed_dict={
                        self.critic_model.input_action_ph: act_for_grad,
                        self.critic_model.input_state_ph: states
                    })

            with tf.variable_scope("train_actor"):
                # Train the actor network with the critic gradients
                self.tf_run_op(
                    [self.actor_model.opt],
                    feed_dict={
                        self.actor_model.input_ph: states,
                        self.actor_model.action_gradients: action_gradient[0]
                    })

            with tf.variable_scope("soft_update"):
                # Update target network
                self._update_target_network()

            # Save weights of the models
            self.actor_saver.save(self.tf_session,
                                  os.path.join(self.actor_model.path,
                                               self.actor_file))
            self.critic_saver.save(self.tf_session,
                                   os.path.join(self.critic_model.path,
                                                self.critic_file))

        def _update_target_network(self, just_copy=False):
            """Update target network with soft update if just_copy is False
            and just copy weights from model to target if True.
            """
            if just_copy:
                self.tf_run_op(self.init_actor_target)
                self.tf_run_op(self.init_critic_target)
            else:
                self.tf_run_op(self.update_actor_target)
                self.tf_run_op(self.update_critic_target)

        def act(self, state):
            """Return action given a state.
            Implements epsilon-greedy exploration.
            """
            # Act randomly if there is not enough samples in the memory
            if len(self.memory) < self.params.warm_up_step:
                return self.act_random()

            # Predict action from state
            reshaped_state = state.reshape(
                1, self.env.observation_space.shape[0])
            feed_A = {self.actor_model.input_ph: reshaped_state}
            action = scale(self.tf_run_op(self.actor_model.output, feed_A),
                           self.act_low, self.act_high)
            if self.params.epsilon_greedy:
                feed_Q = {self.critic_model.input_state_ph: reshaped_state,
                          self.critic_model.input_action_ph: action}
                Q = self.tf_run_op(self.critic_model.Q, feed_Q)[0]
                epsilon = 1 / max(1, Q)
                rand = np.random.rand()
                print("EPSILON, Q, rand:", epsilon, Q, rand)
                if epsilon > rand:
                    return self.act_random()
            return action[0]

        def reset(self):
            super(AC_Policy, self).reset()
            self.noise.reset()
            return self.env.reset()

        def run(self):
            """Run the simulation.
            """
            print(self.params)

            # True to initialize actor and critic with saved weights
            if self.params.load_weights is not None:
                self.actor_saver.restore(self.tf_session,
                                         os.path.join("saved_folder",
                                                      self.initial_name_run,
                                                      self.actor_file))
                self.critic_saver.restore(self.tf_session,
                                          os.path.join("saved_folder",
                                                       self.initial_name_run,
                                                       self.critic_file))

            seed = 42
            np.random.seed(seed)
            tf.set_random_seed(seed)
            self.env.seed(seed)
            state = None

            for j in range(self.params.epochs):
                # Reset the environment if done
                if self.params.reset or state is None:
                    state = self.reset()

                for i in range(self.params.steps):
                    print("EPOCH:", j, "STEP:", i)

                    if self.decaying_noise:
                        noise_scale = (self.params.initial_noise_scale *
                                       self.params.noise_decay ** i) * (
                                           self.act_high - self.act_low)

                    # Render the environment
                    self.render()

                    action = self.act(state)
                    print("ACTION:", action)
                    if j < self.params.noise_threshold_epoch:
                        if self.decaying_noise:
                            action += self.noise() * noise_scale
                        else:
                            action += self.noise()
                            action = np.clip(
                                action, self.act_low, self.act_high)
                    print("ACTION WITH NOISE:", action)

                    new_state, reward, done, _ = self.env.step(action)
                    if new_state[0] < 0.1:
                        done = True

                    print("REWARD:", reward,
                          "MULTIPLY:", self.params.reward_multiply,
                          "DONE:", done)

                    # Put the current environment in the memory
                    # State interval is [-5;5] and action range is [-1;1]
                    self.memory.remember(state, action,
                                         reward * self.params.reward_multiply,
                                         new_state, done)

                    # Don't train if there is not enough samples in the memory
                    if len(self.memory) > self.params.warm_up_step:
                        # Train the network
                        self.train()

                    weights_biases_actor = self.actor_model.summary
                    weights_biases_critic = self.critic_model.summary
                    weights_biases_actor_t = self.target_actor_model.summary
                    weights_biases_critic_t = self.target_critic_model.summary

                    # Plot needed values
                    self.plotting(state=state,
                                  reward=reward,
                                  c_loss=self.critic_loss,
                                  a_loss=self.actor_loss,
                                  epoch=j,
                                  additional=[weights_biases_actor,
                                              weights_biases_critic,
                                              weights_biases_actor_t,
                                              weights_biases_critic_t])

                    if done:
                        break

                    # Change current state
                    state = new_state

            # Save the plot
            self.board.save()
    return AC_Policy
