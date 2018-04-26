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
        super(AC_Policy, self).__init__(args, "Actor Critic algorithm")
        self.memory = Memory()
        self.tf_session = tf.Session()
        self.noise = Noise(mu=np.zeros(self.env.action_space.shape[0]))
        self.actor_file = "actor.h5"
        self.critic_file = "critic.h5"
        
        ################################################################
        # ACTOR
        ################################################################
        self.actor_model = Actor(self.env.observation_space,
                                 self.env.action_space,
                                 self.params.learning_rate,
                                 self.params.tau)
        # Get a second actor model to use it for target and copy the
        # weight of the first actor model to it
        self.target_actor_model = Actor(self.env.observation_space,
                                 self.env.action_space,
                                        self.params.learning_rate,
                                        self.params.tau)
        self._update_target_network(self.actor_model, self.target_actor_model, True)

        ################################################################
        # CRITIC
        ################################################################
        self.critic_model = Critic(self.env.observation_space,
                                   self.env.action_space,
                                   self.params.learning_rate,
                                   self.params.tau)
        # Get a second critic model to use it for target and copy their
        # weight of the first critic model to it 
        self.target_critic_model = Critic(self.env.observationn_space,
                                   self.env.action_space,
                                   self.params.learning_rate)
        self._update_target_network(self.critic_model, self.target_critic_model, True)

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

        states, actions, rewards, next_states, dones = self.memory.samples(self.params.batch_size)

        # Predicted actions
        next_actions = self.tf_session.run(self.actor_model.output,
                                           feed_dict={
                                               self.actor_model.input_h: states
                                           })
        # Compute the Q+1 value with next s+1 and a+1
        Q_next = self.tf_session.run(self.critic_model.Q, feed_dict={
            self.critic_model.input_state_ph: next_actions,
            self.critic_model.input_action_ph: next_states
        })

        Q = rewards + self.params.gamma * Q_next * (1 - dones)

        feed_critic =  {
                self.critic_model.input_state_ph: states,
                self.critic_model.input_action_ph: actions,
                self.critic_model.true_target: Q
            }
        critic_loss, _, critic_action_gradient = self.tf_session.run(
            [self.critic_model.loss, self.critic_model.opt,
             self.critic_model.action_gradients],
            feed_dict=feed_critic)

        feed_actor = {
            self.actor_model.input_ph: states,
            self.actor_model.action_gradients: critic_action_gradient
        }
        self.tf_session.run([self.actor_model.loss, self.actor_model.opt],
                            feed_dict=feed_actor)

        self._update_target_network(self.actor_model, self.target_actor_model)
        self._update_target_network(self.critic_model, self.target_critic_model)

    def _update_target_network(self, model, target, just_copy=False):
        """Update target network with soft update if just_copy is False
        and just copy weights from model to target if True.
        """
        # feed = {
        #     target.from_weights: model.global_variables,
        #     target.just_copy: just_copy
        # }
        # self.tf_session.run(target.update_target, feed)
        pass

    def act(self, state):
        """Return action given a state and add noise to it.
        Implements epsilon-greedy exploration.
        """
        if self.params.train:
            epsilon = self.params.epsilon
            decay = self.params.epsilon_decay
            # Decaying function
            self.params.epsilon = epsilon * ((1 - decay) ** (self.t % self.params.epochs))

            # Return a random action epsilon percent of the time
            if np.random.random() < self.params.epsilon:
                return self.env.action_space.sample()

        reshaped_state = state.reshape(1, self.env.observation_space.shape[0])
        feed = {self.actor_model.input_ph: reshaped_state}
        return self.tf_session.run(self.actor_model.output, feed) + self.noise()

    def run(self):
        """Run the simulation.
        """
        state = self.env.reset()

        # True to initialize actor and critic with saved weights
        if self.params.load_weights:
            self.model_builder.load_model_weights(self.actor_model, self.actor_file)
            self.model_builder.load_model_weights(self.critic_model, self.critic_file)

        for j in range(self.params.epochs):
            action = self.act(state)
            new_state, reward, done, info = self.env.step(action)

            # Put the current environment in the memory
            self.memory.remember(state, action, reward, new_state, done)

            # Reset the environment if done
            self.reset(done)

            # Render the environment
            self.render()

            # Plot needed values
            self.plotting(state, self.params.epsilon, reward)

            # Change current state
            state = new_state

            # Train the network
            self.train()

        # Save the plot
        self.board.save()
