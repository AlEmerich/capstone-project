import matplotlib.pyplot as plt
import numpy as np

plt.ion()
class Board():
    """Board in order to plot some metrics.
    The x axis is the same for all subplots.
    """
    def __init__(self, title):
        """Set the up title of the plot.
        """
        self.title = title

    def on_launch(self, row, column, labels):
        """Initialize the plot.
        :param row: number of rows 
        :param column: number of columns
        :param labels: name of the y axis
        """
        nb_plots = len(labels)
        # Check if there is enough labels for the number of plots
        if row * column != nb_plots:
            raise Exception("row and column you provide is not equal to the number of labels")

        self.row = row
        self.column = column

        # Set up plots
        self.figure, self.ax = plt.subplots(row, column)
        # Set the the up title
        self.figure.suptitle(self.title, fontsize=14, fontweight='bold')

        self.lines = []
        for i in range(self.row):
            for j in range(self.column):
                # Get the lines in a list initilizing it
                self.lines.append(self.ax[i, j].plot([], [])[0])
                # Autoscale botk axis
                self.ax[i, j].set_autoscalex_on(True)
                self.ax[i, j].set_autoscaley_on(True)
                self.ax[i, j].grid()
                # Set the label
                self.ax[i, j].set_ylabel(labels[self._index2dto1d(i, j)])
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                            wspace=0.5)
        plt.ion()

    def on_running(self, ydatas, xdata):
        """Call to update the plots.

        :param ydatas: data to add to the already render data. 
        The order of the data should match the order of the labels
        provided in on_launch()

        :param xdata: the xdata to add to every subplots.
        """
        if(len(ydatas) != self.row * self.column):
            raise Exception("Number of data you want to plot is different than the number of plots")

        for i in range(self.row):
            for j in range(self.column):
                # Get the line
                line = self.lines[self._index2dto1d(i, j)]
                # Add the xdata to the already present one
                line.set_xdata(np.append(line.get_xdata(), xdata))
                # Add the corresponding ydata to the already present one
                line.set_ydata(np.append(line.get_ydata(), ydatas[self._index2dto1d(i, j)]))
                # Recompute the limit
                self.ax[i, j].relim()
                # Autoscale the view
                self.ax[i, j].autoscale_view()

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def on_reset(self):
        """Call when the environment is reset.
        It basically flush every subplots.
        """
        for i in range(self.row):
            for j in range(self.column):
                # Get the line
                line = self.lines[self._index2dto1d(i, j)]
                # Flush datas
                line.set_xdata()
                line.set_ydata()

    def _index2dto1d(self, row, column):
        """Convert X-Y coordinates in one index to 
        acces a 2-D flatten array by the equation 
        `(length of row) * row_index + row_column`

        :Example:

        For a length [3, 3]:
        [1,2] = 3 * 1 + 2 = 5
        """
        return (self.column * row) + column
