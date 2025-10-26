import gymnasium as gym
from gym import spaces
import numpy as np
import graphicsDisplay
import ghostAgents
import multiAgents
import pacman as pacmanFile # assuming pacman.py is in the same folder or properly importable
import layout

class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}


class PacmanEnv(gym.Env):
    def __init__(self, layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
        super(PacmanEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.grid_size = (31, 28)
        self.observation_space = spaces.Box(low=0, high=5, shape=self.grid_size, dtype=np.int8)

        # Load the layout properly
        # self.layout_name = layout
        # self.layout = layout.getLayout(self.layout_name)  # <- use getLayout from layout.py

        # Initialize the game
        self.game = pacmanFile.runGames(layout, pacman, 
                                ghosts,display,
            numGames, record, catchExceptions, timeout)  # Use the game controller class
        self.done = False

    def reset(self):
        # Reset the game
        self.game = pacmanFile.GameState(layout=self.layout)
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        # Convert game state to a numeric array
        # 0 = empty, 1 = wall, 2 = food, 3 = pacman, 4 = ghost
        grid = np.zeros(self.grid_size, dtype=np.int8)
        
        for x, y in self.game.walls:
            grid[y, x] = 1
        
        for food in self.game.food:
            grid[food[1], food[0]] = 2
        
        pacman_pos = self.game.pacmanPosition
        grid[pacman_pos[1], pacman_pos[0]] = 3
        
        for ghost_pos in self.game.getGhostPositions():
            grid[ghost_pos[1], ghost_pos[0]] = 4
        
        return grid

    def step(self, action):
        if self.done:
            raise RuntimeError("Game is over. Call reset() to start a new episode.")

        # Map discrete action to pacman action
        # 0: up, 1: down, 2: left, 3: right
        action_map = {0: 'North', 1: 'South', 2: 'West', 3: 'East'}
        reward = self.game.update(action_map[action])

        self.done = self.game.isOver()
        observation = self._get_observation()

        return observation, reward, self.done, {}

    def render(self, mode="human"):
        self.game.display()

    def close(self):
        pass
