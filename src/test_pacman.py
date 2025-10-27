from gym_env import PacmanEnv
from pacman import readCommand
import sys
from pacmanAgents import GreedyAgent

args = readCommand(sys.argv[1:])

NUM_EPISODES = 3

episode_returns: list[float] = [] # stores each episode's return

for episode in range(NUM_EPISODES):
    env = PacmanEnv(**args)
    obs = env.reset(**args)

    done = False
    step_count = 0

    while not done:
        obs, reward, done, info = env.step()
        # env.render()
        step_count += 1

    env.render(final_reward = reward, episode = episode + 1, num_steps = step_count)
    
    #print("Final reward:", reward)
    episode_returns.append(reward)
    env.close()