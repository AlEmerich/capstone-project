from .abstract_env import AbstractHumanoidEnv
import time

class RandomPolicy(AbstractHumanoidEnv):
    """Class who let the roboschool humanoid
    not learning anything.
    Will be use as benchmark.
    """
    def __init__(self, args):
        """Create the title before calling the super class
        and create the labels of the plot before initializing it.
        """
        super(RandomPolicy, self).__init__(args)
        self.use_matplotlib("Random Policy")

    def act(self, state):
        """ Return a random action.
        """
        return self.env.action_space.sample()

    def run(self, train_pass=10000, epochs=500):
        """Training loop.

        :param train_pass: Number of allowing reset, inf if reset is false.
        :param epochs: Number of maximum loop before reset at each train_pass.
        """
        state = self.env.reset()

        for j in range(epochs):
            state, reward, done, info = self.env.step(self.act(state))

            # Render the environment if asked to
            self.render()

            # Plot the different metrics if asked to
            self.plotting(state=state, reward=reward)

            # Reset the environment when done if asked to
            self.reset(done)
        self.board.save()


if __name__ == "__main__":
    humanoid = RandomPolicy()
    humanoid.run()
