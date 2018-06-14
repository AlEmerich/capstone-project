from abc import ABC
import datetime
import tensorflow as tf
import os


class AbstractActorCritic(ABC):
    """Model manager for actor critic agents. It build models
    and have function to save it.
    """
    def __init__(self, observation_space, action_space,
                 lr, tau, batch_size, scope):
        """Save state space and action space and create
        the folder where to save the weights of the neural network.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.folder = "saved_folder"
        # Get time as string
        sub = str(datetime.datetime.now())
        self.sub_folder = sub.replace(" ", "_")
        self._create_weights_folder(self.folder)
        self._create_weights_folder(
            os.path.join(self.folder, self.sub_folder))
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.scope = scope

        self.init_w = tf.random_normal_initializer(0., 0.3)
        self.init_b = tf.constant_initializer(0.1)

    def _create_weights_folder(self, path):
        """Create the weights folder if not exists."""
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model_weights(self, sess, filepath):
        """Save the weights of thespecified model
        to the specified file in folder specified
        in __init__.
        """
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.folder, filepath))

    def load_model_weights(self, sess, filepath):
        """Load the weights of the specified model
        to the specified file in folder specified
        in __init__.
        """
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.folder, filepath))

    def _soft_update_weights(self, sess, from_weights, just_copy):
        #########################################
        # SOFT TARGET UPDATE
        #########################################

        copy = lambda to_, from_: from_
        soft_update = lambda to_, from_: from_ * self.tau + (
            1 - self.tau) * to_

        fn_ = copy if just_copy else soft_update

        for to_, from_ in zip(self.global_variables, from_weights):
            op_ = tf.assign(to_, fn_(to_, from_), name="soft_update")
            sess.run(op_)

    def _summary_layer(self, name):
        scope = tf.get_variable_scope().name
        var = tf.global_variables(scope=scope+"/"+name)
        weights = var[0]
        biases = var[1]
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)


class Actor(AbstractActorCritic):
    """Policy network.
    """
    def __init__(self, observation_space,
                 action_space,
                 layers,
                 lr,
                 tau,
                 batch_size,
                 scope,
                 dropout,
                 batch_norm,
                 trainable=True):
        """Create the actor model and return it with
        input layer in order to train with the gradients
        of the critic network.
        """
        super(Actor, self).__init__(observation_space,
                                    action_space, lr, tau,
                                    batch_size, scope)

        act = tf.nn.relu

        with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("model"):
                self.input_ph = tf.placeholder(
                    tf.float32, [None, self.observation_space.shape[0]],
                    name='state_input')

                self.action_gradients = tf.placeholder(
                    tf.float32, [None, self.action_space.shape[0]],
                    name='action_gradients')

                with tf.variable_scope("state_input"):
                    h_out = None
                    for i, nb_node in enumerate(layers):
                        prefix = str(nb_node)+"_"+str(i)
                        h_out = tf.layers.dense(
                            h_out if h_out is not None
                            else self.input_ph,
                            units=nb_node,
                            activation=act,
                            kernel_initializer=self.init_w,
                            bias_initializer=self.init_b,
                            name="dense_"+prefix)
                        self._summary_layer("dense_"+prefix)
                        if dropout is not 0:
                            h_out = tf.nn.dropout(h_out, dropout)
                        if batch_norm:
                            h_out = tf.layers.batch_normalization(
                                h_out,
                                name="batch_norm_"+prefix)

                with tf.variable_scope("action_output"):
                    # Action space is from -1 to 1 and it is the range of
                    # hyperbolic tangent
                    self.output = tf.layers.dense(
                        h_out, units=self.action_space.shape[0],
                        activation=tf.nn.tanh,
                        name="output")
                    self._summary_layer("output")

            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope=self.scope+"/model")
            print("ACTOR NETWORK PARAMS IN MEMBER:\n", self.network_params)
            if trainable:
                with tf.variable_scope("train"):
                    self.unnormalized_actor_gradients = tf.gradients(
                        self.output, self.network_params, -self.action_gradients)
                    self.actor_gradients = list(
                        map(
                            lambda x: tf.div(x, self.batch_size),
                            self.unnormalized_actor_gradients))
                    self.loss = tf.reduce_mean(
                        tf.multiply(-self.action_gradients, self.output),
                        name="loss")

                    self.opt = tf.train.AdamOptimizer(
                        self.lr).apply_gradients(
                            zip(self.actor_gradients, self.network_params))


class Critic(AbstractActorCritic):
    """Q network.
    """
    def __init__(self, observation_space,
                 action_space,
                 layers,
                 lr,
                 tau,
                 batch_size,
                 scope,
                 dropout,
                 batch_norm,
                 trainable=True):
        """Create the critic model and return the two input layers
        in order to compute gradients from critic network.
        """
        super(Critic, self).__init__(observation_space,
                                     action_space, lr, tau,
                                     batch_size, scope)

        act = tf.nn.relu

        with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("model"):
                self.true_target_ph = tf.placeholder(
                    tf.float32, [None, 1],
                    name='true_target')

                ################################################################
                # STATE
                ################################################################
                with tf.variable_scope("state_input"):
                    self.input_state_ph = tf.placeholder(
                        tf.float32, [None, self.observation_space.shape[0]],
                        name='state_input')
                    h_state = None
                    for i, nb_node in enumerate(layers):
                        prefix = str(nb_node)+"_"+str(i)
                        h_state = tf.layers.dense(
                            h_state if h_state is not None
                            else self.input_state_ph,
                            units=nb_node,
                            activation=act,
                            kernel_initializer=self.init_w,
                            bias_initializer=self.init_b,
                            name="dense_"+prefix)
                        self._summary_layer("dense_"+prefix)
                        if dropout is not 0:
                            h_state = tf.nn.dropout(h_state, dropout)
                        if batch_norm:
                            h_state = tf.layers.batch_normalization(
                                h_state,
                                name="batch_norm_"+prefix)

                ################################################################
                # ACTION
                ################################################################
                with tf.variable_scope("action_input"):
                    self.input_action_ph = tf.placeholder(
                        tf.float32, [None, self.action_space.shape[0]],
                        name='action_input')
                    h_action = tf.layers.dense(
                        self.input_action_ph,
                        units=layers[-1],
                        activation=act,
                        kernel_initializer=self.init_w,
                        bias_initializer=self.init_b,
                        name="dense_"+str(layers[-1]))

                with tf.variable_scope("Q_output"):
                    merge = h_state + h_action
                    merge = tf.layers.dense(merge, 32, activation=act)
                    l2_reg = tf.contrib.layers.l2_regularizer(scale=0.001)
                    self.Q = tf.layers.dense(merge, 1, activation=None,
                                             kernel_initializer=self.init_w,
                                             bias_initializer=self.init_b,
                                             kernel_regularizer=l2_reg,
                                             name="out")
                    self._summary_layer("out")

            if trainable:
                with tf.variable_scope("gradients"):
                    self.action_gradients = tf.gradients(
                        self.Q, self.input_action_ph, name="action_gradients")

                with tf.variable_scope("train"):
                    self.loss = tf.losses.mean_squared_error(
                        self.true_target_ph, self.Q)
                    self.opt = tf.train.AdamOptimizer(
                        self.lr).minimize(self.loss)
