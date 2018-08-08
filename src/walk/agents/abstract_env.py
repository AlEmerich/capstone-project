from OpenGL import GLU # prevent running error
from abc import ABC, abstractmethod
from ..utils.matplotboard import MatplotBoard
from ..utils.tensorboard import TensorBoard
from collections import namedtuple
import roboschool
import numpy as np
import os
import gym
import sys
import datetime


class AbstractEnv(ABC):

    def __init__(self, args, sub_folder=None):
        self.params = namedtuple("Params", args.keys())(*args.values())

        # list of rewards updating at each step
        self.rewards = []

        self.t = 0
        # save the t from the latest reset
        self.last_t = 0
        # list of t corresponding to the life time of the env
        self.list_reset_t = []
        self.board = None

        self.folder = "saved_folder"

        if sub_folder is None:
            sub = str(datetime.datetime.now())
            sub_folder = sub.replace(" ", "_")

        if not os.path.exists(self.folder):
            sys.exit("Please create the " + self.folder + " folder manually.")

    def _create_weights_folder(self, path, count=0):
        """Create the weights folder if not exists."""
        r_path = path + "_" + str(count)
        if os.path.exists(r_path):
            return self._create_weights_folder(path, count+1)
        else:
            os.makedirs(r_path)
        return r_path

    @abstractmethod
    def plotting(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, done):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def run(self):
        pass


class AbstractMountainCarEnv(AbstractEnv, ABC):
    def __init__(self, args, name_run):
        super(AbstractMountainCarEnv, self).__init__(args)
        # When training, we are not interested at the same metrics
        # than when not training
        self.labels = None
        if self.params.train:
            self.labels = ["Reward", "Average reward",
                           "Critic loss", "Actor loss"]
        else:
            self.labels = ["Average reward"]
        self.env = gym.make("MountainCarContinuous-v0")

        # Definition of the observation and action space
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        self.name_run = name_run
        self.saved_folder = self._create_weights_folder(
                os.path.join(self.folder, self.name_run))
        print("FOLDER WEIGHTS:", self.saved_folder)

    def use_matplotlib(self, title):
        """Set the agent to use matplotboard to plot metrics.
        """
        self.board = MatplotBoard(title, self.name_run)
        self.board.on_launch(row=2, column=3, labels=self.labels)

    def use_tensorboard(self, tf_session):
        """Set the agent to use tensorboard to plot metrics.
        """
        self.board = TensorBoard(tf_session, self.name_run)
        self.board.on_launch(labels=self.labels)

    def plotting(self, **kwargs):
        """Send values to plot to render it.
        Actually plot the total reward, the distance to the target,
        the distance of the grivity center from the ground and the
        angle to the target.
        """
        if self.board:

            # Unpack kwargs*
            reward = kwargs.get('reward')
            c_loss = kwargs.get('c_loss')
            a_loss = kwargs.get('a_loss')
            train = kwargs.get('train')
            # increment t and add the reward to the list
            self.t += 1
            self.rewards.append(reward)

            # Define text to display at the center of the figure
            info = None
            if self.list_reset_t:
                info = ' '.join(["RESET at t =", str(self.last_t),
                                 ", Best t so far:",
                                 str(max(self.list_reset_t)),
                                 ", Average t :",
                                 str(np.average(self.list_reset_t))])

            # Must match the order of the labels defined in __init__
            ydatas = None
            if train:
                ydatas = [reward, np.average(self.rewards),
                          c_loss, a_loss]
            else:
                ydatas = [np.average(self.rewards)]

            # Data to plot in the Y axis of the subplots
            self.board.on_running(ydatas=ydatas,
                                  xdata=self.t,
                                  info=info)

    def reset(self):
        """Reset the time value and the total reward
        and reset the gym environment
        """
        if self.params.reset:
            # Add the life time of the finished session to the list
            current_t = self.t - self.last_t
            self.list_reset_t.append(current_t)
            self.last_t = self.t

            # reset the list of reward to average with new values
            self.rewards = []

            # reset env and plot red line on subplots to indicate reset
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
    def run(self):
        pass


class AbstractHumanoidEnv(AbstractEnv, ABC):
    """ Super class of all policy.
    """

    def __init__(self, args, name_run):
        """Instantiate Roboschool humanoid environment and reset it.

        :param args: arguments of the program to send to the Params.
        """
        super(AbstractHumanoidEnv, self).__init__(args)

        # When training, we are not interested at the same metrics
        # than when not training
        self.labels = None

        self.initial_name_run = name_run
        if self.params.train:
            self.labels = ["Reward", "Average reward",
                           "Distance gravity center from ground",
                           "Critic loss", "Actor loss"]
            self.name_run = name_run + "_train"
        else:
            self.labels = ["Average reward", "Angle to target",
                           "Distance to target", "Gravity center from ground"]
            self.name_run = name_run + "_run"
        self.env = gym.make("RoboschoolHumanoid-v1")

        # Defintion of observation and action space
        self.obs_low = np.empty(len(self.env.observation_space.low))
        self.obs_low.fill(-5)        # Define in robotschool environment
        self.obs_high = np.empty(len(self.env.observation_space.high))
        self.obs_high.fill(5)
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        self.saved_folder = self._create_weights_folder(
                os.path.join(self.folder, self.name_run))
        print("FOLDER WEIGHTS:", self.saved_folder)

    def use_matplotlib(self, title):
        """Set the agent to use matplotboard to plot metrics.
        """
        self.board = MatplotBoard(title, self.name_run)
        self.board.on_launch(row=2, column=3, labels=self.labels)

    def use_tensorboard(self, tf_session):
        """Set the agent to use tensorboard to plot metrics.
        """
        self.board = TensorBoard(tf_session, self.name_run)
        self.board.on_launch(labels=self.labels)

    def plotting(self, **kwargs):
        """Send values to plot to render it.
        Actually plot the total reward, the distance to the target,
        the distance of the grivity center from the ground and the
        angle to the target.
        """
        if self.board:

            # Unpack kwargs
            state = kwargs.get('state')
            reward = kwargs.get('reward')
            c_loss = kwargs.get('c_loss')
            a_loss = kwargs.get('a_loss')
            epoch = kwargs.get('epoch')

            # Get metrics
            target_dist = self.env.unwrapped.walk_target_dist
            dist_center_ground = state[0]
            angle_to_target = self.env.unwrapped.angle_to_target

            # increment t and add the reward to the list
            self.t += 1
            self.rewards.append(reward)

            # Define text to display at the center of the figure
            info = None
            if self.list_reset_t:
                info = ' '.join(["RESET at t =", str(self.last_t),
                                 ", Best t so far:",
                                 str(max(self.list_reset_t)),
                                 ", Average t:",
                                 str(np.average(self.list_reset_t)),
                                 ", Epoch:", str(epoch)])

            # Must match the order of the labels defined in __init__
            ydatas = None
            if self.params.train:
                ydatas = [reward, np.average(self.rewards),
                          dist_center_ground, c_loss, a_loss]
            else:
                ydatas = [np.average(self.rewards), angle_to_target,
                          target_dist, dist_center_ground]

            # Data to plot in the Y axis of the subplots
            self.board.on_running(ydatas=ydatas,
                                  xdata=self.t,
                                  info=info)

    def reset(self):
        """Reset the time value and the total reward
        and reset the gym environment
        """
        if self.params.reset:
            # Add the life time of the finished session to the list
            current_t = self.t - self.last_t
            self.list_reset_t.append(current_t)
            self.last_t = self.t

            # reset the list of reward to average with new values
            self.rewards = []

            # reset env and plot red line on subplots to indicate reset
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
    def run(self):
        pass
