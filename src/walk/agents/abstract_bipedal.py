from abc import ABC, abstractmethod
from .abstract_env import AbstractEnv
from ..utils.matplotboard import MatplotBoard
from ..utils.tensorboard import TensorBoard
import os
import gym
import numpy as np


class AbstractBipedalEnv(AbstractEnv, ABC):
    def __init__(self, args, name_run):
        super(AbstractBipedalEnv, self).__init__(args)
        # When training, we are not interested at the same metrics
        # than when not training
        self.labels = None
        if self.params.train:
            self.labels = ["Reward", "Average reward",
                           "Critic loss", "Actor loss"]
            self.name_run = name_run + "_train"

        else:
            self.labels = ["Reward", "Average reward"]
            self.name_run = name_run + "_run"

        self.env = gym.make("BipedalWalker-v2")

        self.initial_name_run = name_run

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
            additional = kwargs.get('additional')

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
            if self.params.train:
                ydatas = [reward, np.average(self.rewards),
                          c_loss, a_loss]
            else:
                ydatas = [reward, np.average(self.rewards)]

            # Data to plot in the Y axis of the subplots
            self.board.on_running(ydatas=ydatas,
                                  xdata=self.t,
                                  additional=additional,
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

            # reset env and plot red line on subplots to indicate reset
            self.board.on_reset(self.t, self.rewards)

            # reset the list of reward to average with new values
            self.rewards = []

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
