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

    def _variable(self, name, shape, dim, init=None, reg=None):
        if init is None:
            init = tf.random_uniform(
                shape,
                -1/math.sqrt(dim), 1/math.sqrt(dim))
        return tf.get_variable(
            name=name,
            initializer=init,
            regularizer=reg
        )

    def _fc_layer(self, layer, nb_input,
                  units, w_init=None, b_init=None,
                  w_reg=None, b_reg=None):
        W = self._variable(
            "W"+layer,
            [nb_input, units],
            nb_input,
            init=w_init,
            reg=w_reg
        )
        self.summary.append(tf.summary.histogram("layer"+layer+"/weights", W))
        B = self._variable(
            "B"+layer,
            [units],
            nb_input,
            init=b_init,
            reg=b_reg
        )
        self.summary.append(tf.summary.histogram("layer"+layer+"/biases", B))
        return W, B

    def _summary_layer(self, name):
        scope = tf.get_variable_scope().name
        var = tf.global_variables(scope=scope+"/"+name)
        weights = var[0]
        biases = var[1]
        self.summary.append(
            tf.summary.histogram(name+"/weights", weights))
        self.summary.append(
            tf.summary.histogram(name+"/biases", biases))
