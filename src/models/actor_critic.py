import keras
from ..utils.memory import Memory

class ActorCritic():

    def __init__(self, env, batch_size=36):
        self.env = env
        self.memory = Memory()

        self.batch_size = batch_size

    def actor_model():
        pass

    def critic_model():
        pass
