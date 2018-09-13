from OpenGL import GLU # prevent running error
from abc import ABC, abstractmethod
from .abstract_env import AbstractEnv
from ..utils.matplotboard import MatplotBoard
from ..utils.tensorboard import TensorBoard
import os
import numpy as np
import gym, roboschool


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
            additional = kwargs.get('additional')

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

    def benchmark(self):
        print(self.params)

        seed = 42
        np.random.seed(seed)
        self.env.seed(seed)
        state = None

        for j in range(self.params.epochs):
            # Reset the environment if done
            if self.params.reset or state is None:
                state = self.reset()

            for i in range(self.params.steps):
                print("EPOCH:", j, "STEP:", i)

                # Render the environment
                self.render()

                action = self.act_random()

                new_state, reward, done, _ = self.env.step(action)

                # Plot needed values
                self.plotting(state=state,
                              c_loss=0,
                              a_loss=0,
                              reward=reward,
                              epoch=j)

                if done:
                    break

                # Change current state
                state = new_state
        self.board.save()

    def act_random(self):
        return self.env.action_space.sample()

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def run(self):
        pass
