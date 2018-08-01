import roboschool, gym
import matplotlib.pyplot as plt

env = gym.make("RoboschoolHumanoid-v1")

EPOCHS = 500

s = env.reset()

x = []
y = []

for i in range(EPOCHS):
    s = list(env.step(env.action_space.sample()))
    y.append(s[0])
    x.append(i)

for xe, ye in zip(x, y):
    plt.scatter([xe] * len(ye), ye)
plt.show()
