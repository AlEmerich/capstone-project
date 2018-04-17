import matplotlib.pyplot as plt
import numpy as np

plt.ion()
class Board():

    def __init__(self, title):
        self.title = title

    def on_launch(self, row, column, labels):
        nb_plots = len(labels)
        if row * column != nb_plots:
            raise Exception("row and column you provide is not equal to the number of labels")

        self.row = row
        self.column = column

        #Set up plot
        self.figure, self.ax = plt.subplots(row, column)
        self.figure.suptitle(self.title, fontsize=14, fontweight='bold')
        self.lines = []
        for i in range(self.row):
            for j in range(self.column):
                self.lines.append(self.ax[i, j].plot([], [])[0])
                self.ax[i, j].set_autoscalex_on(True)
                self.ax[i, j].set_autoscaley_on(True)
                self.ax[i, j].grid()
                self.ax[i, j].set_ylabel(labels[self._index2dto1d(i, j)])
        plt.ion()

    def on_running(self, ydatas, xdata):

        if(len(ydatas) != self.row * self.column):
            raise Exception("Number of data you want to plot is different than the number of plots")

        for i in range(self.row):
            for j in range(self.column):
                line = self.lines[self._index2dto1d(i, j)]
                line.set_xdata(np.append(line.get_xdata(), xdata))
                line.set_ydata(np.append(line.get_ydata(), ydatas[self._index2dto1d(i, j)]))
                self.ax[i, j].relim()
                self.ax[i, j].autoscale_view()

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def on_reset(self):
        for i in range(self.row):
            for j in range(self.column):
                line = self.lines[self._index2dto1d(i, j)]
                line.set_xdata()
                line.set_ydata()

    def _index2dto1d(self, row, column):
        return (self.column * row) + column
