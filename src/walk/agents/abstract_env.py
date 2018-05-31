from OpenGL import GLU # prevent running error
from abc import ABC, abstractmethod
from ..utils.params import Params
from ..utils.matplotboard import MatplotBoard
from ..utils.tensorboard import TensorBoard
from collections import namedtuple
import numpy as np
import roboschool
import gym


class AbstractHumanoidEnv(ABC):
    """ Super class of all policy.
    """

    def __init__(self, args):
        """Instantiate Roboschool humanoid environment and reset it.

        :param args: arguments of the program to send to the Params.
        """
        # self.params = Params(args)
        self.params = namedtuple("Params", args.keys())(*args.values())
        print()
        # list of rewards updating at each step
        self.rewards = []

        self.t = 0
        # save the t from the latest reset
        self.last_t = 0
        # list of t corresponding to the life time of the env
        self.list_reset_t = []
        self.board = None

        # When training, we are not interested at the same metrics
        # than when not training
        self.labels = None
        if self.params.train:
            self.labels = ["Reward", "Average reward",
                           "Distance gravity center from ground",
                           "Epsilon", "Critic loss", "Actor loss"]
        else:
            self.labels = ["Average reward", "Angle to target",
                           "Distance to target", "Gravity center from ground"]
        self.env = gym.make("RoboschoolHumanoid-v1")

    def use_matplotlib(self, title):
        """Set the agent to use matplotboard to plot metrics.
        """
        self.board = MatplotBoard(title)
        self.board.on_launch(row=2, column=3, labels=self.labels)

    def use_tensorboard(self, tf_session):
        """Set the agent to use tensorboard to plot metrics.
        """
        self.board = TensorBoard(tf_session)
        self.board.on_launch(labels=self.labels)

    def plotting(self, **kwargs):
        """Send values to plot to render it.
        Actually plot the total reward, the distance to the target,
        the distance of the grivity center from the ground and the
        angle to the target.

        :param state: The current state of the environment.
        :param reward: The reward to plot.
        """
        if self.board:
            # Unpack kwargs
            state = kwargs.get('state')
            reward = kwargs.get('reward')
            epsilon = kwargs.get('epsilon')
            c_loss = kwargs.get('c_loss')
            a_loss = kwargs.get('a_loss')

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
                          epsilon, c_loss, a_loss]
            else:
                ydatas = [np.average(self.rewards), angle_to_target,
                          target_dist, dist_center_ground]

            # Define text to display at the center of the figure
            info = None
            if self.list_reset_t:
                info = ' '.join(["RESET at t =", str(self.last_t),
                                 ", Best t so far:",
                                 str(max(self.list_reset_t)),
                                 ", Average t :",
                                 str(np.average(self.list_reset_t))])

            # Data to plot in the Y axis of the subplots
            self.board.on_running(ydatas=ydatas,
                                  xdata=self.t,
                                  info=info)

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
