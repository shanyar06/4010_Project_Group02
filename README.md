**Introduction:** 

This project applies Reinforcement Learning into our custom version of Pacman. In our case the agent learns how to navigate through a grid world maze, eat food pellets, and eat a ghost. The environment is designed to emulate an agent and gym-compatible environment interaction in a visual way in order to experiment with algorithms such as Q-Learning and Policy Iteration.

**Game Rules:**
- The Pac-Man attempts to eat all pellets while avoiding ghosts. Both the Pac-Man and the ghosts start in the same location.
- We run multiple episodes. And each episode ends when either:
    - Pac-Man runs into a ghost (loses the game)
    - Pac-Man consumes all the pellets (wins the game)
    - The time limit/timeout occurs
- We also have the following scoring systems:
    - +250 for Winning the game (eating last food) 
    - -250 for Losing the game (getting caught by a ghost)
    - +100 for Eating a ghost
    - +1 for Eating a food pellet
    - -1 for Dithering (not eating anything)

**Purpose:**
Our project's purpose is to implement a custom gym environment with an agent that can use any RL algorithm. We also aim to demonstrate how the agent interacts with a visual, dynamic environment by using the reset() and step() functions. These functions are responsible for initializing a new game state, or representing a new game state. We also aim to create a good testing ground that can be used for different RL algorithms and policies. Finally, we would like to encourage Pac-Man to complete the game efficiently and successfully, with few losses as time progresses.

**Instructions to Run:**
- Make sure you have a version of Python 3.12+ installed (this can be checked by running "python --version" in terminal). Also make sure you have pip installed on your machine.
- In the root folder, run "pip install -r requirements.txt"
- Then navigate into the "src" directory
- We can run multiple episodes via the following command: "python src/test_pacman.py -p GreedyAgent -l smallClassic --frameTime 0.1"
- The following are the command-line arguments specified:
    - -p: Pac-Man agent type (ex. GreedyAgent, RandomAgent)
    - -l: The layout to use (ex. smallClassic, mediumClassic)
    - --frameTime: Affects the speed of the display

**Referenced Repo**
The Github repository we're referencing implements an Approximate Q-Learning approach to train a Pacman agent. The implementation uses Iterative Deepening Search (IDS) to measure ghost proximity more accurately than Manhattan distance and defines a straightforward reward function, assigning large penalties for hitting active ghosts and positive rewards for eating scared ghosts, food pellets, or winning the game. The model was trained on the mediumClassic map and achieved up to 90–100% win rates after about 200 games. However, the project can be improved in several ways. We plan to improve it by replacing the linear model 
