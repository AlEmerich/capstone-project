from abc import ABC, abstractmethod
import datetime
import os
import sys


class Board(ABC):
    """Board in order to plot some metrics.
    The x axis is the same for all subplots.
    """
    def __init__(self, name_run):
        """Set the up title of the plot.
        """
        folder = "logs"

        if not os.path.exists(folder):
            sys.exit("Please create the " + folder + " folder manually")
        self.path = self._create_folder(os.path.join(folder, name_run))

    def _create_folder(self, folder, count=0):
        """Create folder if it doesn't exists.
        """
        r_path = folder + "_" + str(count)
        if os.path.exists(r_path):
            return self._create_folder(folder, count=count+1)
        else:
            os.makedirs(r_path)
        return r_path

    @abstractmethod
    def on_launch(self, **kwargs):
        """Initialize the board.
        """
        pass

    @abstractmethod
    def on_running(self, **kwargs):
        """Call to update the board.
        """
        pass

    @abstractmethod
    def on_reset(self, t):
        """Call when the environment is reset.
        """
        pass
