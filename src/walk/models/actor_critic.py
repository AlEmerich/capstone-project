from abc import ABC
import tensorflow as tf


class AbstractActorCritic(ABC):
    """Model manager for actor critic agents. It build models
    and have function to save it.
    """
    def __init__(self, observation_space, action_space,
                 lr, tau, batch_size, scope, path):
        """Save state space and action space and create
        the folder where to save the weights of the neural network.
        """
        self.observation_space = observation_space
        self.action_space = action_space

        self.lr = lr

        self.tau = tau
        self.batch_size = batch_size
        self.scope = scope

        self.path = path

        self.init_w_u = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.init_w_n = tf.truncated_normal_initializer(-3e-3, 3e-3)
        self.init_b = tf.constant_initializer(0.0)

        self.summary = []

    def _summary_layer(self, name):
        scope = tf.get_variable_scope().name
        var = tf.global_variables(scope=scope+"/"+name)
        weights = var[0]
        biases = var[1]
        self.summary.append(
            tf.summary.histogram(name+"/weights", weights))
        self.summary.append(
            tf.summary.histogram(name+"/biases", biases))


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
                 path,
                 trainable=True):
        """Create the actor model and return it with
        input layer in order to train with the gradients
        of the critic network.
        """
        super(Actor, self).__init__(observation_space,
                                    action_space, lr, tau,
                                    batch_size, scope,
                                    path)

        act = tf.nn.relu
        output_act = tf.nn.tanh
        # l2_reg = tf.contrib.layers.l2_regularizer(scale=0.001)

        with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("model"):
                self.input_ph = tf.placeholder(
                    tf.float32, [None, self.observation_space.shape[0]],
                    name='state_input')

                self.action_gradients = tf.placeholder(
                    tf.float32, [None, self.action_space.shape[0]],
                    name='action_gradients')

                with tf.variable_scope("hidden_input"):
                    h_out = None
                    for i, nb_node in enumerate(layers):
                        prefix = str(nb_node)+"_"+str(i)
                        h_out = tf.layers.dense(
                            h_out if h_out is not None
                            else self.input_ph,
                            units=nb_node,
                            activation=act,
                            kernel_initializer=self.init_w_n,
                            bias_initializer=self.init_b,
                            name="dense_"+prefix)
                        self._summary_layer("dense_"+prefix)
                        if dropout is not 0:
                            h_out = tf.nn.dropout(h_out, dropout)
                        if batch_norm:
                            h_out = tf.layers.batch_normalization(
                                h_out,
                                training=trainable,
                                name="batch_norm_"+prefix)

                with tf.variable_scope("action_output"):
                    # Action space is from -1 to 1 and it is the range of
                    # hyperbolic tangent
                    self.output = tf.layers.dense(
                        h_out, units=self.action_space.shape[0],
                        activation=output_act,
                        kernel_initializer=self.init_w_u,
                        bias_initializer=self.init_b,
                        name="output")
                    self._summary_layer("output")

            self.network_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.scope+"/model")

            if trainable:
                with tf.variable_scope("train"):
                    self.actor_gradients = tf.gradients(
                        self.output, self.network_params,
                        self.action_gradients)
                    # self.actor_gradients = list(
                    #    map(
                    #        lambda x: tf.div(x, self.batch_size),
                    #        self.unnormalized_actor_gradients))
                    self.opt = tf.train.AdamOptimizer(
                        -self.lr).apply_gradients(
                            zip(self.actor_gradients,
                                self.network_params))


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
                 path,
                 trainable=True):
        """Create the critic model and return the two input layers
        in order to compute gradients from critic network.
        """
        super(Critic, self).__init__(observation_space,
                                     action_space, lr, tau,
                                     batch_size, scope,
                                     path)

        act = tf.nn.relu
        l2_reg = tf.contrib.layers.l2_regularizer(scale=0.001)

        with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("model"):
                self.true_target_ph = tf.placeholder(
                    tf.float32, [None, 1],
                    name='true_target')

                ##############################################
                # STATE
                ##############################################
                with tf.variable_scope("state_input"):
                    self.input_state_ph = tf.placeholder(
                        tf.float32, [None, self.observation_space.shape[0]],
                        name='state_input')
                    h_state = None
                    for i, nb_node in enumerate(layers[:-1]):
                        prefix = str(nb_node)+"_"+str(i)
                        h_state = tf.layers.dense(
                            h_state if h_state is not None
                            else self.input_state_ph,
                            units=nb_node,
                            activation=act,
                            kernel_initializer=self.init_w_n,
                            bias_initializer=self.init_b,
                            name="dense_"+prefix)
                        self._summary_layer("dense_"+prefix)
                        if dropout is not 0:
                            h_state = tf.nn.dropout(h_state, dropout)
                        if batch_norm:
                            h_state = tf.layers.batch_normalization(
                                h_state,
                                training=trainable,
                                name="batch_norm_"+prefix)

                ###############################################
                # ACTION
                ###############################################
                with tf.variable_scope("action_input"):
                    self.input_action_ph = tf.placeholder(
                        tf.float32, [None, self.action_space.shape[0]],
                        name='action_input')

                with tf.variable_scope("merging"):
                    h_action = tf.layers.dense(
                        self.input_action_ph,
                        layers[-1],
                        kernel_initializer=self.init_w_n,
                        name="dense_action_"+str(layers[-1]))

                    h_state = tf.layers.dense(
                        h_state,
                        layers[-1],
                        kernel_initializer=self.init_w_n,
                        use_bias=False,
                        name="dense_state_"+str(layers[-1]))

                with tf.variable_scope("Q_output"):
                    merge = act(h_state + h_action)
                    # merge = tf.layers.dense(merge, layers[-1],
                    #                         activation=act,
                    #                         kernel_initializer=self.init_w_n,
                    #                         bias_initializer=self.init_b,
                    #                         name="dense_"+str(layers[-1]))
                    # self._summary_layer("dense_"+str(layers[-1]))
                    if batch_norm:
                        merge = tf.layers.batch_normalization(
                            merge,
                            training=trainable,
                            name="batch_norm_merge")
                    if dropout is not 0:
                        merge = tf.nn.dropout(merge, dropout)

                    self.Q = tf.layers.dense(merge, 1, activation=None,
                                             kernel_initializer=self.init_w_u,
                                             bias_initializer=self.init_b,
                                             kernel_regularizer=l2_reg,
                                             name="out")
                    self._summary_layer("out")

            self.network_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.scope+"/model")

            if trainable:
                with tf.variable_scope("gradients"):
                    self.action_gradients = tf.gradients(
                        self.Q, self.input_action_ph, name="action_gradients")

                with tf.variable_scope("train"):
                    self.loss = tf.losses.mean_squared_error(
                        self.true_target_ph, self.Q)
                    self.opt = tf.train.AdamOptimizer(
                        self.lr).minimize(self.loss)
