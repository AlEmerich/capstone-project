import tensorflow as tf
from .actor_critic import AbstractActorCritic


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

        self.act = tf.nn.relu
        self.output_act = tf.nn.tanh

        with tf.variable_scope("actor"):
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
            self.input_ph = tf.placeholder(
                tf.float32, [None, self.observation_space.shape[0]],
                name='state_input')

            # HIDDEN
            W1, B1 = self._fc_layer(
                "1", self.observation_space.shape[0], layers[0])
            W2, B2 = self._fc_layer(
                "2", layers[0], layers[1])

            # OUTPUT
            W3, B3 = self._fc_layer(
                "3", layers[1], self.action_space.shape[0],
                w_init=tf.random_uniform(
                    [layers[1], self.action_space.shape[0]],
                    -3e-3,
                    3e-3),
                b_init=tf.random_uniform(
                    [self.action_space.shape[0]],
                    -3e-3,
                    3e-3)
            )

            if not batch_norm:
                layer1 = self.act(tf.matmul(self.input_ph, W1) + B1)
                layer2 = self.act(tf.matmul(layer1, W2) + B2)
                self.output = self.output_act(tf.matmul(layer2, W3) + B3)
            else:
                layer1 = self.act(tf.matmul(self.input_ph, W1) + B1)
                bn1 = tf.layers.batch_normalization(layer1)
                layer2 = self.act(tf.matmul(bn1, W2) + B2)
                bn2 = tf.layers.batch_normalization(layer2)
                self.output = self.output_act(tf.matmul(bn2, W3) + B3)
        return [W1, B1, W2, B2, W3, B3]

    def _create_training_function(self):
        """ Create the training tensorflow operation.
        """
        with tf.variable_scope("train"):
            self.action_gradients = tf.placeholder(
                tf.float32, [None, self.action_space.shape[0]],
                name='action_gradients')

            self.actor_gradients = tf.gradients(
                self.output, self.network_params,
                -self.action_gradients)

            printed = list(
                map(lambda x:
                    tf.Print(x,
                             [tf.clip_by_norm(
                                 tf.div(x, self.batch_size), 1.0)],
                             "Grads " + str(x.shape)),
                    self.actor_gradients))

            self.opt = tf.train.AdamOptimizer(
                self.lr).apply_gradients(
                    zip(printed,
                        self.network_params))
