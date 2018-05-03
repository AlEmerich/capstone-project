from abc import ABC, abstractmethod
import datetime
import os


class Board(ABC):
    """Board in order to plot some metrics.
    The x axis is the same for all subplots.
    """
    def __init__(self):
        """Set the up title of the plot.
        """
        folder = "plots"
        now = str(datetime.datetime.now())
        sub_folder = now.replace(" ", "_")

        self._create_folder(folder)
        self.path = os.path.join(folder, sub_folder)
        self._create_folder(self.path)

    def _create_folder(self, folder):
        """Create folder if it doesn't exists.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

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
