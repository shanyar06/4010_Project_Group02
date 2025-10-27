# Main game controller in order to run our Pac-Man gym environment.
# Uses the gym__env.PacmanEnv

import time
import numpy as np
import argparse

# importing readCommand from pacman. This is used for parsing command line arguments and builds a dictionary (args) in order to run the Pac-Man game
from pacman import readCommand

# importing the gym_env
from gym_env import PacmanEnv

# Making a policy class. Our agent will use a epsilon greedy policy where epsilon is close to 1 to promote exploration
class Policy:
    def __init__(self, action_space, epsilon: float =1.0, seed: int | None = None):
        self.random_num_gen = np.random.default_rng(seed) # sets up an rng using seed
        self.action_space = action_space # keeps track of all the actions, it will consist of up, right, down, and left
        self.epsilon = float(epsilon) # keeps track of epsilon (as that determines how much randomness and exploration occurs)

    def action_to_perform(self, observation):
        # returns an action for the agent to perform. If epsilon stays as the default value (1.0) then the action will be random.
        # uses an observation since the environment must tell the agent what it sees right now for the agent to decide what to do. 

        # performing a random action (as of right now this is entered every time since the agent performs a random action)
        # we can easily use a Q-table for this later
        if self.random_num_gen.random() < self.epsilon:
            return self.random_num_gen.integers(0, self.action_space.n) # choosing a random action to perform (going up, right, down, or left)

        # otherwise, using argmax over Q(observation, a) 
        # TODO: implement Q-learning for the policy
        return self.random_num_gen.integers(0, self.action_space.n)  # TODO: Replace with the greedy behaviour


def run(episodes: int = 5, max_num_steps: int = 1000, seed: int = 101231382, render: bool = False, pacman_cli: list[str] | None = None, epsilon: float = 1.0):
    #Inputs:
        # episodes: how many episodes we should run (5 by default)
        # max_num_steps: limit on how many steps to take per action
        # seed: Seed for the RNG (default value is to use Pallav's student number (101231382) for reproducability)
        # render: whether we should call env.render() at each step or not.
        # pacman_cli: list of CLI passed by the user to readCommand() from pacman.py (this includes the agent, layout, display, etc.)
        # epsilon: for epsilon-greedy exploration
    
    # If the user passes no CLI to readCommand, then pacman_cli will be an empty array (for readCommand)
    if pacman_cli is None: 
        pacman_cli = []

    # parsing command line arguments and builds a dictionary (args) in order to run the Pac-Man game using readCommand
    pacman_game_args = readCommand(pacman_cli)

    # building the environment and policy (using epsilon-greedy policy)
    env = PacmanEnv(**pacman_game_args)
    policy = Policy(env.action_space, epsilon, seed)

    episode_returns: list[float] = [] # stores each episode's return
    successes = 0 # success counter
    t0 = time.time() # starting the timer

    # Loop for each episode
    for episode in range(1, episodes + 1):
        # Trying to reset the environment
        try:
            observation = env.reset()
        except TypeError:
            observation, info = env.reset()

        # keeping track of step count and total for each episode.
        done = False # boolean flag for if we are done
        total = 0.0
        steps = 0
        
        # Step loop
        while steps < max_num_steps and not done:
            if render:
                # if render is true, trying to render the environment. Otherwise, if we are unable to render then skipping the render
                try:
                    env.render()
                except Exception:
                    pass
            
            action = policy.action_to_perform(observation) # choosing an action to perform based on the policy
            observation, reward, done, info = env.step(action) # advancing to the next time step

            # updating the total based on the reward and incrementing step
            total += float(reward)
            steps += 1
        
        # After the step loop is complete, storing the episode's return to the episode_returns list
        episode_returns.append(total)

        # If the total is >0, then considering it a success
        if total > 0:
            successes += 1

        print(f"EPISODE {episode}: Return={total}  Steps={steps}  Done={done}")

    # Displaying the average return, the rate of how many episodes are successful, and how long it took
    duration = time.time() - t0
    
    if episode_returns:
        avg_return = float(np.mean(episode_returns))
    else:
        avg_return = 0.0

    success_rate = successes / episodes

    print("---------------")
    print("EPISODE SUMMARY")
    print("---------------")

    print(f"There were {episodes} episodes")
    print(f"Average return was {avg_return:.2f}")
    print(f"Success rate was {success_rate*100}%")
    print(f"Total duration {duration:.2f} seconds")

    # trying to close the environment
    try:
        env.close()
    except Exception:
        pass

def main():
    argument_parser = argparse.ArgumentParser(description = "Pac-Man Env Demo Main Game Controller") # creating an arg parser in order to read command-line inputs like

    # flags for episodes, max_num_steps, seed, epsilon, and render
    argument_parser.add_argument("--episodes", type=int, default=5, help="How many episodes to run")
    argument_parser.add_argument("--max_num_steps", type=int, default=1000, help="Step limit per episode")
    argument_parser.add_argument("--seed", type=int, default=101231382, help="RNG seed")
    argument_parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for epsilon-greedy")
    argument_parser.add_argument("--render", choices=["on", "off"], default="off", help="Render each step")

    # Collecting extra arguments in pacman_cli
    known, pacman_cli = argument_parser.parse_known_args()

    run(episodes=known.episodes, max_num_steps=known.max_num_steps, seed=known.seed, render=(known.render == "on"), pacman_cli=pacman_cli, epsilon=known.epsilon)

if __name__ == "__main__":
    main()