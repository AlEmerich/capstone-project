import tensorflow as tf
from .actor_critic import AbstractActorCritic


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

        self.act = tf.nn.relu
        self.l2_reg = tf.contrib.layers.l2_regularizer(scale=0.001)

        with tf.variable_scope("critic"):
            self.network_params = self._create_network(layers, batch_norm)

            if trainable:
                self._create_training_function()

    def _create_network(self, layers, batch_norm):
        """ Build the network.
        :params layers: the list of layers to build
        :params batch_norm: boolean, to add batch normalization or not.
        :return the created variables.
        """

        with tf.variable_scope("model"):
            with tf.variable_scope("placeholder"):
                self.true_target_ph = tf.placeholder(
                    tf.float32, [None, 1],
                    name='true_target')
                self.input_state_ph = tf.placeholder(
                    tf.float32, [None, self.observation_space.shape[0]],
                    name='state_input')
                self.input_action_ph = tf.placeholder(
                    tf.float32, [None, self.action_space.shape[0]],
                    name='action_input')

            # STATE
            W1, B1 = self._fc_layer("1",
                                    self.observation_space.shape[0],
                                    layers[0])

            # MERGING ACTIONS
            W2 = self._variable(
                "W2",
                [layers[0], layers[1]],
                layers[0] + self.action_space.shape[0],
                reg=self.l2_reg
            )
            self.summary.append(
                tf.summary.histogram("layer2/weights", W2))
            W2_action = self._variable(
                "W2_action",
                [self.action_space.shape[0], layers[1]],
                layers[0] + self.action_space.shape[0],
                reg=self.l2_reg
            )
            self.summary.append(
                tf.summary.histogram("layer_action/weights", W2_action))
            B2 = self._variable(
                "B2",
                [layers[1]],
                layers[0] + self.action_space.shape[0],
            )
            self.summary.append(
                tf.summary.histogram("layer2/biases", B2))

            # OUTPUT
            W3, B3 = self._fc_layer(
                "3", layers[1], 1,
                w_init=tf.random_uniform(
                    [layers[1], 1],
                    -3e-3,
                    3e-3),
                w_reg=self.l2_reg,
                b_init=tf.random_uniform(
                    [1],
                    -3e-3,
                    3e-3))

            if not batch_norm:
                layer1 = self.act(tf.matmul(self.input_state_ph, W1) + B1)
                layer2 = self.act(tf.matmul(layer1, W2) +
                                  tf.matmul(self.input_action_ph,
                                            W2_action) +
                                  B2)
                self.Q = tf.identity(tf.matmul(layer2, W3) + B3)
            else:
                layer1 = self.act(tf.matmul(self.input_state_ph, W1) + B1)
                bn1 = tf.layers.batch_normalization(layer1)
                layer2 = self.act(tf.matmul(bn1, W2) +
                                  tf.matmul(self.input_action_ph,
                                            W2_action) +
                                  B2)
                bn2 = tf.layers.batch_normalization(layer2)
                self.Q = tf.identity(tf.matmul(bn2, W3) + B3)

        return [W1, B1, W2, W2_action, B2, W3, B3]

    def _create_training_function(self):
        """ Create the training tensorflow operation.
        """
        with tf.variable_scope("gradients"):
            self.action_gradients = tf.gradients(
                self.Q, self.input_action_ph, name="action_gradients")

        with tf.variable_scope("train"):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.true_target_ph, self.Q)) + \
                tf.losses.get_regularization_loss()
            self.opt = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)
