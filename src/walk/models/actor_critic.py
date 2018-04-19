from keras.layers import Input, Dense, concatenate
from keras.models import Model
from ..utils.memory import Memory

class ActorCritic():

    def actor_model(self, env):
        input_layer = Input(shape=self.env.observation_space.shape)

        h1 = Dense(32, activation="relu")(input_layer)
        h2 = Dense(64, activation="relu")(h1)
        h3 = Dense(32, activation="relu")(h2)

        output = Dense(self.env.action_space.space, activation="tanh")(h3)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def critic_model(self, env):
        input_state = Input(shape=self.env.observation_space.shape)
        h1 = Dense(32, activation="relu")(input_state)

        input_action = Input(shape=self.env.action_space.shape)
        h2 = Dense(32, activation="relu")(input_action)

        merge = concatenate([h1, h2])
        out_1 = Dense(64, activation="relu")(merge)
        out_2 = Dense(128, activation="relu")(out_1)
        out_3 = Dense(64, activation="relu")(out_2)
        out_4 = Dense(32, activation="relu")(out_3)

        model = Model(inputs=[input_state, input_action], outputs=out_4)
        model.compile(optimizer="adam", loss="mse")
        return model
