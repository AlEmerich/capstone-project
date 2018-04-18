from OpenGL import GLU # prevent running error
from abc import ABC, abstractmethod
from ..utils.params import Params
from ..utils.board import Board
import numpy as np
import matplotlib.pyplot as plt
import roboschool, gym

class AbstractHumanoidEnv(ABC):
    """ Super class of all policy.
    """

    def __init__(self, args):
        """Instantiate Roboschool humanoid environment and reset it.

        :param args: arguments of the program to send to the Params.
        """
        self.params = Params(args)
        self.total_reward = 0
        self.t = 0
        # Title has to be define in child class
        self.board = Board(self.title)
        self.env = gym.make("RoboschoolHumanoid-v1")
        self.env.reset()

    def plotting(self, state, reward):
        """Send values to plot to render it. 
        Actually plot the total reward, the distance to the target,
        the distance of the grivity center from the ground and the
        angle to the target. 

        :param state: The current state of the environment.
        :param reward: The reward to plot.
        """
        target_dist = self.env.unwrapped.walk_target_dist
        dist_center_ground = state[0]
        angle_to_target = self.env.unwrapped.angle_to_target

        # increment t and the total reward
        self.t += 1
        self.total_reward += reward

        ydatas = [self.total_reward, target_dist, dist_center_ground, angle_to_target]
        # Data to plot in the Y axis of the subplots
        self.board.on_running(ydatas, self.t)

    def reset(self):
        """Reset the time value and the total reward
        and reset the gym environment
        """
        self.t = 0
        self.total_reward = 0
        self.env.reset()
        self.board.on_reset()

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def train(self):
        pass