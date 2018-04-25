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
        # list of rewards updating at each step
        self.rewards = []

        self.t = 0
        # save the t from the latest reset
        self.last_t = 0
        # list of t corresponding to the life time of the env
        self.list_reset_t = []

        # Title has to be define in child class
        self.board = Board(title)

        # When training, we are not interested at the same metrics
        # than when not training
        labels = None
        if self.params.train:
            labels = ["Reward", "Average reward",
                      "Number of epochs", "Epsilon"]
        else:
            labels = ["Average reward", "Angle to target",
                      "Distance to target", "Gravity center from ground"]

        self.board.on_launch(row=2, column=2, labels=labels)
        self.env = gym.make("RoboschoolHumanoid-v1")

    def plotting(self, state, epsilon, reward):
        """Send values to plot to render it.
        Actually plot the total reward, the distance to the target,
        the distance of the grivity center from the ground and the
        angle to the target.

        :param state: The current state of the environment.
        :param reward: The reward to plot.
        """
        if self.params.plot:

            # Get metrics
            target_dist = self.env.unwrapped.walk_target_dist
            dist_center_ground = state[0]
            angle_to_target = self.env.unwrapped.angle_to_target

            # increment t and add the reward to the list
            self.t += 1
            self.rewards.append(reward)

            # Must match the order of the labels defined in __init__
            ydatas = None
            if self.params.train:
                ydatas = [reward, np.average(self.rewards),
                          dist_center_ground,
                          epsilon]
            else:
                ydatas = [np.average(self.rewards), angle_to_target,
                          target_dist, dist_center_ground]

            # Define text to display at the center of the figure
            info = None
            if self.list_reset_t:
                info = ' '.join(["RESET at t =", str(self.last_t),
                                 ", Best t so far:", str(max(self.list_reset_t)),
                                 ", Average t :", str(np.average(self.list_reset_t))])

            # Data to plot in the Y axis of the subplots
            self.board.on_running(ydatas, self.t, info=info)

    def reset(self, done):
        """Reset the time value and the total reward
        and reset the gym environment
        """
        if done and self.params.reset:
            # Add the life time of the finished session to the list
            current_t = self.t - self.last_t
            self.list_reset_t.append(current_t)
            self.last_t = self.t

            # reset the list of reward to average with new values
            self.rewards = []

            # reset env and plot red line on subplots to indicate reset
            self.env.reset()
            self.board.on_reset(self.t)

    def render(self):
        """render the environment if asked to
        """
        if self.params.render:
            self.env.render()

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def run(self, train_pass, epochs):
        pass
