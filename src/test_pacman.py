from gym_env import PacmanEnv
from pacman import readCommand
import sys

args = readCommand(sys.argv[1:])

env = PacmanEnv(**args)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
