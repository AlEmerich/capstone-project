from abstract_env import AbstractHumanoidEnv
import time

class RandomPolicy(AbstractHumanoidEnv):
    """Class who let the roboschool humanoid
    not learning anything.
    Will be use as benchmark.
    """

    def act(self):
        """ Return a random action
        """
        return self.env.action_space.sample()

    def train(self, epochs=10000):
        try:
            while 1:
                # time.sleep(0.1)
                for j in range(epochs):
                    print("step")
                    state, reward, done, info = self.env.step(self.act())
                    print(state)
                    print("render")
                    #self.env.render()
                    if done:
                        print("reset")
                        observation = self.env.reset()
                        print(observation)
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interruption")


from OpenGL import GLU # prevent running error
import roboschool, gym

if __name__ == "__main__":
    humanoid = RandomPolicy()
    humanoid.train()
