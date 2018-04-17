from .abstract_env import AbstractHumanoidEnv
import time

class RandomPolicy(AbstractHumanoidEnv):
    """Class who let the roboschool humanoid
    not learning anything.
    Will be use as benchmark.
    """
    def __init__(self, args):
        self.title = "Random policy"
        super(RandomPolicy, self).__init__(args)
        labels = ["Reward", "Distance to target", "Gravity center from ground", "Angle to target"]
        self.board.on_launch(row=2, column=2, labels=labels)

    def act(self):
        """ Return a random action
        """
        return self.env.action_space.sample()

    def train(self, epochs=10000):
        try:
            while 1:
                # time.sleep(0.1)
                for j in range(epochs):
                    state, reward, done, info = self.env.step(self.act())
                    if self.params.render:
                        self.env.render()
                    if done:
                        if self.params.reset:
                            observation = self.reset()
                    print("*****************************",
                          self.env.unwrapped.walk_target_dist,
                          "*****************************")
                    self.plotting(state, reward)
        except KeyboardInterrupt:
            print("Keyboard interruption")


if __name__ == "__main__":
    humanoid = RandomPolicy()
    humanoid.train()
