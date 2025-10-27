from gym_env import PacmanEnv
from pacman import readCommand
import sys
from pacmanAgents import GreedyAgent

args = readCommand(sys.argv[1:])

env = PacmanEnv(**args)

obs = env.reset(**args)

done = False

while not done:
    obs, reward, done, info = env.step()
    # env.render()
print("Final reward:", reward)

env.close()