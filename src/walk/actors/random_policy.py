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
        self.title = "Random policy"
        super(RandomPolicy, self).__init__(args)
        labels = ["Reward", "Distance to target", "Gravity center from ground", "Angle to target"]
        self.board.on_launch(row=2, column=2, labels=labels)

    def act(self):
        """ Return a random action.
        """
        return self.env.action_space.sample()

    def train(self, train_pass=10, epochs=10000):
        """Training loop.
        
        :param train_pass: Number of allowing reset, inf if reset is false.
        :param epochs: Number of maximum loop before reset at each train_pass.
        """
        try:
            for i in range(train_pass):
                for j in range(epochs):
                    state, reward, done, info = self.env.step(self.act())

                    # Render the environment if asked to
                    if self.params.render:
                        self.env.render()

                    # Plot the different metrics if asked to
                    if self.params.plot:
                        self.plotting(state, reward)

                    # Reset the environment when done if asked to
                    if done and self.params.reset:
                            observation = self.reset()
                            break
        except KeyboardInterrupt:
            print("Keyboard interruption")


if __name__ == "__main__":
    humanoid = RandomPolicy()
    humanoid.train()
