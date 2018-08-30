from abc import ABC, abstractmethod
from collections import namedtuple
import os
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
