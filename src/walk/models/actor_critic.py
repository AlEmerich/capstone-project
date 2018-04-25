from keras.layers import Input, Dense, concatenate
from keras.layers import BatchNormalization, Dropout, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from ..utils.memory import Memory
import os

class ActorCritic():
    """Model manager for actor critic agents. It build models
    and have function to save it.
    """
    def __init__(self, observation_space, action_space):
        """Save state space and action space and create
        the folder where to save the weights of the neural network.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.folder = "saved_folder"
        self._create_weights_folder(self.folder)

    def _create_weights_folder(self, path):
        """Create the weights folder if not exists."""
        if not os.path.exists(path):
            os.makedirs(path)

    def actor_model(self, params):
        """Create the actor model and return it with
        input layer in order to train with the gradients
        of the critic network.
        """
        input_layer = Input(shape=self.observation_space.shape)

        h_out = input_layer
        for nb_node in [256, 128, 64, 32]:
            h_dense = Dense(nb_node)(h_out)
            batch_norm = BatchNormalization()(h_dense)
            activation = Activation('relu')(batch_norm)
            h_out = activation

        # Action space is from -1 to 1 and it is the range of
        # hyperbolic tangent
        output = Dense(self.action_space.shape[0], activation="tanh")(h_out)
        model = Model(inputs=input_layer, outputs=output)
        opt = Adam(lr=params.learning_rate)
        model.compile(optimizer=opt, loss='mse')
        model.summary()
        return input_layer, model

    def critic_model(self, params):
        """Create the critic model and return the two input layers
        in order to compute gradients from critic network.
        """
        input_state = Input(shape=self.observation_space.shape)

        state_out = input_state
        for nb_node in (256, 128, 64, 32):
            hidden = Dense(nb_node, activation="relu")(state_out)
            batch_norm = BatchNormalization()(hidden)
            activation = Activation('relu')(batch_norm)
            state_out = activation

        input_action = Input(shape=self.action_space.shape)

        action_out = input_action
        for nb_node in (128, 64):
            hidden = Dense(64, activation="relu")(action_out)
            action_out = hidden

        merge = concatenate([state_out, action_out])
        out_1 = Dense(64, activation="relu")(merge)
        out_2 = Dense(32, activation="relu")(out_1)
        out_3 = Dense(16, activation="relu")(out_2)
        out_4 = Dense(1, activation="relu")(out_3)

        model = Model(inputs=[input_state, input_action], outputs=out_4)
        opt = Adam(lr=params.learning_rate)
        model.compile(optimizer=opt, loss="mse")
        model.summary()
        return input_state, input_action, model

    def save_model_weights(self, model, filepath):
        """Save the weights of thespecified model
        to the specified file in folder specified
        in __init__.
        """
        model.save_weights(os.path.join(self.folder, filepath))

    def load_model_weights(self, model, filepath):
        """Load the weights of the specified model
        to the specified file in folder specified
        in __init__.
        """
        model.load_weights(os.path.join(self.folder, filepath))

    def callbacks(self, filepath):
        """Defines Keras callback. Just checkpoint for now to save
        the best only.
        """
        checkpoint = ModelCheckpoint(os.path.join(self.folder, filepath),
                                     save_best_only=True, monitor="loss")
        return [checkpoint]
