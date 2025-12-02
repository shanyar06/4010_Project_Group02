from matplotlib import pyplot as plt
from gym_env import PacmanEnv
from pacman import readCommand
import sys
from pacmanAgents import GreedyAgent, TabularQAgent, ApproximateQAgent

args = readCommand(sys.argv[1:])
agent = args['pacman']

NUM_EPISODES = 3

episode_returns: list[float] = [] # stores each episode's return

for episode in range(NUM_EPISODES):
    env = PacmanEnv(**args)
    #obs = env.reset(**args)
    obs = env.reset(
        args['layout'],
        args['pacman'],
        args['ghosts'],
        args['display'],
        args['numGames'],
        args['record'],
        catchExceptions=args.get('catchExceptions', False),
        timeout=args.get('timeout', 30)
    )

    done = False
    step_count = 0
    state = obs

    while not done:
        next_state, reward, done, info = env.step()
        state = next_state
        # env.render()
        step_count += 1

    if hasattr(agent, "final"):
        try:
            agent.final(state)
        except AttributeError:
            pass

    env.render(final_reward = reward, episode = episode + 1, num_steps = step_count)
    
    #print("Final reward:", reward)
    episode_returns.append(reward)
    env.close()

# episodes = list(range(1, len(episode_returns) + 1))

# plt.figure(figsize=(8, 5))
# plt.plot(episodes, episode_returns, marker='o')
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Reward per Episode")
# plt.grid(True)
# plt.show()