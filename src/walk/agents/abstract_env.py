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

    def __init__(self, args, title):
        """Instantiate Roboschool humanoid environment and reset it.

        :param args: arguments of the program to send to the Params.
        """
        self.params = Params(args)
        self.rewards = []
        self.t = 0
        # Title has to be define in child class
        self.board = Board(self.title)
        labels = ["Reward", "Distance to target", "Gravity center from ground", "Angle to target"]
        self.board.on_launch(row=2, column=2, labels=labels)
        self.env = gym.make("RoboschoolHumanoid-v1")

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
        self.rewards.append(reward)

        ydatas = [np.average(self.rewards), target_dist, dist_center_ground, angle_to_target]
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
    def act(self, state):
        pass

    @abstractmethod
    def run(self, train_pass, epochs):
        pass
