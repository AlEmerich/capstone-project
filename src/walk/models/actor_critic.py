from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from ..utils.memory import Memory
import os

class ActorCritic():

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.folder = "saved_folder"
        self._create_weights_folder(self.folder)

    def _create_weights_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def actor_model(self):
        input_layer = Input(shape=self.observation_space.shape)

        h1 = Dense(32, activation="relu")(input_layer)
        h2 = Dense(64, activation="relu")(h1)
        h3 = Dense(32, activation="relu")(h2)

        output = Dense(self.action_space.shape[0], activation="tanh")(h3)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return input_layer, model

    def critic_model(self):
        input_state = Input(shape=self.observation_space.shape)
        h1 = Dense(32, activation="relu")(input_state)

        input_action = Input(shape=self.action_space.shape)
        h2 = Dense(32, activation="relu")(input_action)

        merge = concatenate([h1, h2])
        out_1 = Dense(64, activation="relu")(merge)
        out_2 = Dense(32, activation="relu")(out_1)
        out_3 = Dense(16, activation="relu")(out_2)
        out_4 = Dense(1, activation="relu")(out_3)

        model = Model(inputs=[input_state, input_action], outputs=out_4)
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        return input_state, input_action, model

    def save_model_weights(self, model, filepath):
        model.save_weights(os.path.join(self.folder, filepath))

    def load_model_weights(self, model, filepath):
        model.load_weights(os.path.join(self.folder, filepath))

    def callbacks(self, filepath):
        checkpoint = ModelCheckpoint(os.path.join(self.folder, filepath),
                                     save_best_only=True, monitor="loss")
        return [checkpoint]
