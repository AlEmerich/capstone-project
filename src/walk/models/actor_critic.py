from abc import ABC
import tensorflow as tf
import os

class AbstractActorCritic(ABC):
    """Model manager for actor critic agents. It build models
    and have function to save it.
    """
    def __init__(self, observation_space, action_space,
                 lr, tau):
        """Save state space and action space and create
        the folder where to save the weights of the neural network.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.folder = "saved_folder"
        self._create_weights_folder(self.folder)
        self.lr = lr
        self.tau = tau

    def _create_weights_folder(self, path):
        """Create the weights folder if not exists."""
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model_weights(self, model, filepath):
        """Save the weights of thespecified model
        to the specified file in folder specified
        in __init__.
        """
        pass

    def load_model_weights(self, model, filepath):
        """Load the weights of the specified model
        to the specified file in folder specified
        in __init__.
        """
        pass

    def _soft_update_weights(self):
        #########################################
        # SOFT TARGET UPDATE
        #########################################
        # self.global_variables = tf.get_collection(
        #     tf.GraphKeys.GLOBAL_VARIABLES)

        # self.just_copy = tf.placeholder(tf.bool)

        # soft_update = lambda to_, from_: from_ * self.tau + (
        #     1 - self.tau) * to_

        # self.update_target = [
        #     tf.assign(to, soft_update(to_, from_)) for to, from_
        #     in zip(self.global_variables, self.from_weights)
        # ]
        pass

class Actor(AbstractActorCritic):
    """Policy network.
    """
    def __init__(self, observation_space, action_space, lr, tau):
        """Create the actor model and return it with
        input layer in order to train with the gradients
        of the critic network.
        """
        super(Actor, self).__init__(observation_space,
                                    action_space, lr, tau)

        self.input_ph = tf.placeholder(tf.float32,
            [None, self.observation_space.shape[0]])

        self.action_gradients = tf.placeholder(tf.float32,
            [None, self.action_space.shape[0]])

        h_out = None
        for nb_node in [256, 128, 64, 32]:
            h_out = tf.layers.dense(h_out if h_out is not None
                                    else self.input_ph,
                                    units=nb_node, activation=tf.nn.relu)
            h_out = tf.layers.batch_normalization(h_out)

        # Action space is from -1 to 1 and it is the range of
        # hyperbolic tangent
        self.output = tf.layers.dense(h_out,
                                      units=self.action_space.shape[0],
                                      activation=tf.nn.tanh)

        self.loss = tf.reduce_mean(
            tf.multiply(-self.action_gradients, self.output))

        self.actor_opt = tf.train.AdamOptimizer(
            self.lr).minimize(self.loss)


class Critic(AbstractActorCritic):
    """Q network.
    """
    def __init__(self, observation_space, action_space, lr, tau):
        """Create the critic model and return the two input layers
        in order to compute gradients from critic network.
        """
        super(Critic, self).__init__(observation_space,
                                     action_space, lr, tau)

        self.true_target_ph = tf.placeholder(tf.float32)

        ################################################################
        # STATE
        ################################################################
        self.input_state_ph = tf.placeholder(tf.float32,
            [None, self.observation_space.shape[0]])

        state_out = None
        for nb_node in [256, 128, 64, 32]:
            state_out = tf.layers.dense(state_out if state_out is not None
                                        else self.input_state_ph,
                                        units=nb_node, activation=tf.nn.relu)
            state_out = tf.layers.batch_normalization(state_out)

        ################################################################
        # ACTION
        ################################################################
        self.input_action_ph = tf.placeholder(tf.float32,
            [None, self.action_space.shape[0]])

        action_out = None
        for nb_node in [128, 64]:
            action_out = tf.layers.dense(action_out if action_out is not None
                                         else self.input_action_ph,
                                         units=nb_node, activation=tf.nn.relu)
            action_out = tf.layers.batch_normalization(action_out)

        merge = tf.concat(state_out, action_out)

        out_1 = tf.layers.dense(merge, 64, activation=tf.nn.relu)
        out_2 = tf.layers.dense(out_1, 32, activation=tf.nn.relu)
        out_3 = tf.layers.dense(out_2, 16, activation=tf.nn.relu)
        self.Q = tf.layers.dense(out_3, 1, activation=tf.nn.relu)

        self.action_gradients = tf.gradients(self.Q, self.critic_input_action)

        self.loss = tf.losses.mean_squared_error(
            self.true_target, self.Q)
        self.opt = tf.train.AdamOptimizer(
            self.lr).minimize(self.loss)
