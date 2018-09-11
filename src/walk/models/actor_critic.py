from abc import ABC
import tensorflow as tf
import math


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

    def _variable(self, shape, dim, name):
        return tf.Variable(tf.random_uniform(
            shape, -1/math.sqrt(dim), 1/math.sqrt(dim)), name=name)

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

        with tf.variable_scope("actor"):
            with tf.variable_scope("model"):
                self.input_ph = tf.placeholder(
                    tf.float32, [None, self.observation_space.shape[0]],
                    name='state_input')

                W1 = self._variable(
                    [self.observation_space.shape[0], layers[0]],
                    self.observation_space.shape[0], name="W1")
                self.summary.append(tf.summary.histogram("layer1/weights", W1))
                B1 = self._variable(
                    [layers[0]],
                    self.observation_space.shape[0], name="B1")
                self.summary.append(tf.summary.histogram("layer1/biases", B1))
                W2 = self._variable(
                    [layers[0], layers[1]],
                    layers[0], name="W2")
                self.summary.append(tf.summary.histogram("layer2/weights", W2))
                B2 = self._variable(
                    [layers[1]],
                    layers[0], name="B2")
                self.summary.append(tf.summary.histogram("layer2/biases", B2))
                W3 = tf.Variable(
                    tf.random_uniform(
                        [layers[1], self.action_space.shape[0]],
                        -3e-3,
                        3e-3), name="W3")
                self.summary.append(tf.summary.histogram("output/weights", W3))
                B3 = tf.Variable(
                    tf.random_uniform(
                        [self.action_space.shape[0]],
                        -3e-3,
                        3e-3), name="B3")
                self.summary.append(tf.summary.histogram("output/biases", B3))

            if not batch_norm:
                layer1 = act(tf.matmul(self.input_ph, W1) + B1)
                layer2 = act(tf.matmul(layer1, W2) + B2)
                self.output = output_act(tf.matmul(layer2, W3) + B3)
            else:
                layer1 = act(tf.matmul(self.input_ph, W1) + B1)
                bn1 = tf.layers.batch_normalization(layer1)
                layer2 = act(tf.matmul(bn1, W2) + B2)
                bn2 = tf.layers.batch_normalization(layer2)
                self.output = output_act(tf.matmul(bn2, W3) + B3)
            self.network_params = [W1, B1, W2, B2, W3, B3]

            if trainable:
                with tf.variable_scope("train"):
                    self.action_gradients = tf.placeholder(
                        tf.float32, [None, self.action_space.shape[0]],
                        name='action_gradients')

                    self.actor_gradients = tf.gradients(
                        self.output, self.network_params,
                        -self.action_gradients)
                    # self.actor_gradients = list(
                    #    map(
                    #        lambda x: tf.div(x, self.batch_size),
                    #        self.actor_gradients))
                    self.opt = tf.train.AdamOptimizer(
                        self.lr).apply_gradients(
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

        with tf.variable_scope("critic"):
            with tf.variable_scope("model"):
                self.true_target_ph = tf.placeholder(
                    tf.float32, [None, 1],
                    name='true_target')
                self.input_state_ph = tf.placeholder(
                        tf.float32, [None, self.observation_space.shape[0]],
                        name='state_input')
                self.input_action_ph = tf.placeholder(
                        tf.float32, [None, self.action_space.shape[0]],
                        name='action_input')

                W1 = self._variable(
                        [self.observation_space.shape[0], layers[0]],
                        self.observation_space.shape[0], name="W1")
                self.summary.append(tf.summary.histogram("layer1/weights", W1))
                B1 = self._variable(
                        [layers[0]],
                        self.observation_space.shape[0], name="B1")
                self.summary.append(tf.summary.histogram("layer1/biases", B1))
                W2 = self._variable(
                        [layers[0], layers[1]],
                        layers[0] + self.action_space.shape[0], name="W2")
                self.summary.append(tf.summary.histogram("layer2/weights", W2))
                W2_action = self._variable(
                    [self.action_space.shape[0], layers[1]],
                    layers[0] + self.action_space.shape[0], name="W2_action")
                self.summary.append(
                    tf.summary.histogram("layer_action/weights", W2_action))
                B2 = self._variable(
                    [layers[1]],
                    layers[0] + self.action_space.shape[0], name="B2")
                self.summary.append(tf.summary.histogram("layer2/biases", B2))
                W3 = tf.Variable(
                    tf.random_uniform(
                        [layers[1], 1],
                        -3e-3,
                        3e-3), name="W3")
                self.summary.append(tf.summary.histogram("layer3/weights", W3))
                B3 = tf.Variable(
                    tf.random_uniform(
                        [1],
                        -3e-3,
                        3e-3), name="B3")
                self.summary.append(tf.summary.histogram("layer3/biases", B3))

                if not batch_norm:
                    layer1 = act(tf.matmul(self.input_state_ph, W1) + B1)
                    layer2 = act(tf.matmul(layer1, W2) +
                                 tf.matmul(self.input_action_ph, W2_action) +
                                 B2)
                    self.Q = tf.identity(tf.matmul(layer2, W3) + B3)
                else:
                    layer1 = act(tf.matmul(self.input_state_ph, W1) + B1)
                    bn1 = tf.layers.batch_normalization(layer1)
                    layer2 = act(tf.matmul(bn1, W2) +
                                 tf.matmul(self.input_action_ph, W2_action) +
                                 B2)
                    bn2 = tf.layers.batch_normalization(layer2)
                    self.Q = tf.identity(tf.matmul(bn2, W3) + B3)

            self.network_params = [W1, B1, W2, W2_action, B2, W3, B3]

            if trainable:
                with tf.variable_scope("gradients"):
                    self.action_gradients = tf.gradients(
                        self.Q, self.input_action_ph, name="action_gradients")

                with tf.variable_scope("train"):
                    self.loss = tf.losses.mean_squared_error(
                        self.true_target_ph, self.Q) # / self.batch_size
                    self.opt = tf.train.AdamOptimizer(
                        self.lr).minimize(self.loss)
