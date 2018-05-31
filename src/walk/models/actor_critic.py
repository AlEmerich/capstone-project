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
        var = tf.trainable_variables(scope=scope+"/"+name)
        weights = var[0]
        biases = var[1]
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)


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
                 batch_norm):
        """Create the actor model and return it with
        input layer in order to train with the gradients
        of the critic network.
        """
        super(Actor, self).__init__(observation_space,
                                    action_space, lr, tau,
                                    batch_size, scope)

        act = tf.nn.relu

        with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
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

            with tf.variable_scope("train"):
                self.network_params = tf.trainable_variables(scope=self.scope)
                print("actor params:", len(self.network_params))
                self.actor_gradients = tf.gradients(
                    self.output, self.network_params, -self.action_gradients)

                self.loss = tf.reduce_mean(
                    tf.multiply(self.action_gradients, self.output),
                    name="loss")

                self.opt = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss)


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
                 batch_norm):
        """Create the critic model and return the two input layers
        in order to compute gradients from critic network.
        """
        super(Critic, self).__init__(observation_space,
                                     action_space, lr, tau,
                                     batch_size, scope)

        act = tf.nn.relu

        with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
            self.true_target_ph = tf.placeholder(
                tf.float32, name='true_target')

            ################################################################
            # STATE
            ################################################################
            with tf.variable_scope("state_input"):
                self.input_state_ph = tf.placeholder(
                    tf.float32, [None, self.observation_space.shape[0]],
                    name='state_input')

                # state_out = None
                # for nb_node in [64, 32]:
                #     dense_name = "dense_"+str(nb_node)
                #     state_out = tf.layers.dense(
                #         state_out if state_out is not None
                #         else self.input_state_ph,
                #         units=nb_node,
                #         activation=act,
                #         name=dense_name)
                #     self._summary_layer(dense_name)
                #     state_out = tf.layers.batch_normalization(
                #         state_out,
                #         name="batch_norm_"+str(nb_node))

            ################################################################
            # ACTION
            ################################################################
            with tf.variable_scope("action_input"):
                self.input_action_ph = tf.placeholder(
                    tf.float32, [None, self.action_space.shape[0]],
                    name='action_input')


            with tf.variable_scope("Q_output"):
                # merge = tf.concat([self.input_state_ph, self.input_action_ph],
                #                   axis=1,
                #                   name="merge")
                h_out = None
                for i, nb_node in enumerate(layers):
                    prefix = str(nb_node)+"_"+str(i)
                    h_out = tf.layers.dense(
                        h_out if h_out is not None
                        else self.input_state_ph,
                        units=nb_node,
                        activation=act,
                        name="dense_"+prefix)
                    self._summary_layer("dense_"+prefix)
                    if dropout is not 0:
                        h_out = tf.nn.dropout(h_out, dropout)
                    if batch_norm:
                        h_out = tf.layers.batch_normalization(
                            h_out,
                            name="batch_norm_"+prefix)

                merge = tf.concat([h_out, self.input_action_ph],
                                  axis=1,
                                  name="merge")
                self.Q = tf.layers.dense(merge, 1, activation=tf.nn.tanh,
                                         name="out")
                self._summary_layer("out")

            with tf.variable_scope("train"):
                self.loss = tf.losses.mean_squared_error(
                    self.Q, self.true_target_ph, reduction=tf.losses.Reduction.MEAN)
                self.opt = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss)

            with tf.variable_scope("gradients"):
                self.action_gradients = tf.gradients(
                    self.Q, self.input_action_ph)
