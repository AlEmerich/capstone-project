from .abstract_env import AbstractHumanoidEnv
from ..models.actor_critic import ActorCritic
from ..utils.memory import Memory
from ..utils.noise import Noise
import numpy as np
import keras.backend as K
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
        K.set_session(self.tf_session)
        self.noise = Noise(mu=np.zeros(self.env.action_space.shape[0]))
        self.actor_file = "actor.h5"
        self.critic_file = "critic.h5"
        self.model_builder = ActorCritic(self.env.observation_space,
                                         self.env.action_space)

        ################################################################
        # ACTOR
        ################################################################
        self.actor_input, self.actor_model = self.model_builder.actor_model(self.params)
        # Get a second actor model to use it for target and copy the
        # weight of the first actor model to it
        _, self.target_actor_model = self.model_builder.actor_model(self.params)
        self._update_target_network(self.actor_model, self.target_actor_model, True)

        # Define the placeholder for the gradients
        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.params.batch_size,
                                                 self.env.action_space.shape[0]])
        # Get the weight of the model needed to compute the gradients
        actor_model_weights = self.actor_model.trainable_weights

        # Define the gradients computation
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        # Train applying the gradients
        self.actor_optimizer = tf.train.AdamOptimizer(
            self.params.learning_rate).apply_gradients(grads)

        ################################################################
        # CRITIC
        ################################################################
        self.critic_state_input, self.critic_action_input,\
            self.critic_model = self.model_builder.critic_model(self.params)
        # Get a second critic model to use it for target and copy their
        # weight of the first critic model to it 
        _, _, self.target_critic_model = self.model_builder.critic_model(self.params)
        self._update_target_network(self.critic_model, self.target_critic_model, True)

        # Compute the gradients in order to give it to the actor
        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)
        # Initialize all global variables of tensorflow
        self.tf_session.run(tf.global_variables_initializer())

    def _train_actor(self, states, actions, rewards, new_states, dones):
        """Train the actor, the policy network, against the critic
        gradients.
        """
        # Get next actions from states
        next_actions = self.actor_model.predict_on_batch(states)

        # Compute gradients from the critics
        grads = self.tf_session.run(self.critic_grads,
                                    feed_dict = {
                                        self.critic_state_input: states,
                                        self.critic_action_input: actions
                                    })

        # Train the actor with states and gradients
        self.tf_session.run(self.actor_optimizer,
                            feed_dict = {
                                self.actor_input: states,
                                self.actor_critic_grad: grads
                            })
        # Save the model 
        self.model_builder.save_model_weights(self.actor_model, self.actor_file)


    def _train_critic(self, states, actions, rewards, new_states, dones):
        """Train the critic network in a keras way with fit.
        """
        # predict next actions
        next_actions = self.target_actor_model.predict_on_batch(states)

        # Compute the Q+1 value with next s+1 and a+1
        Q_values_p1 = self.target_critic_model.predict(
            [new_states, next_actions])[0][0]
        # Compute the expected reward with Bellman equation
        future_rewards = rewards + self.params.gamma * Q_values_p1 * (1 - dones)
        # Train the critic network with expected rewards as targets
        self.critic_model.fit([states, actions], future_rewards,
                              verbose=0,
                              callbacks=self.model_builder.callbacks(self.critic_file))

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

        states, actions, rewards, new_states, dones = self.memory.samples(self.params.batch_size)

        # Let's try first to run samples on critic then on actor
        # And after one sample per sample
        self._train_critic(states, actions, rewards, new_states, dones)
        self._train_actor(states, actions, rewards, new_states, dones)

        self._update_target_network(self.actor_model, self.target_actor_model)
        self._update_target_network(self.critic_model, self.target_critic_model)

    def _update_target_network(self, model, target, just_copy=False):
        """Update target network with soft update if just_copy is False
        and just copy weights from model to target if True.
        """
        weights = model.get_weights()
        target_weights = target.get_weights()

        soft_update = lambda x, y: x * self.params.tau + (1 - self.params.tau) * y

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] if just_copy else soft_update(
                target_weights[i], weights[i])

        target.set_weights(target_weights)

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
        return self.actor_model.predict(reshaped_state)[0] + self.noise()

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
