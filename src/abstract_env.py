from OpenGL import GLU # prevent running error
import roboschool, gym
from abc import ABC, abstractmethod

class AbstractHumanoidEnv(ABC):
    """ Super class of all policy
    """

    def __init__(self):
        """Instantiate Roboschool humanoid environment and reset it
        """
        self.env = gym.make("RoboschoolHumanoid-v1")
        self.env.reset()

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def train(self):
        pass
