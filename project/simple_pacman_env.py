import gym
from gym import spaces
import numpy as np
import random

class SimplePacmanEnv(gym.Env):
    """
    Minimal Pacman environment from scratch.
    Grid-based, random ghost moves, simple rewards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(5, 5), num_ghosts=1):
        super().__init__()

        self.grid_size = grid_size
        self.num_ghosts = num_ghosts

        # Actions: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Observation: grid with integers
        # 0: empty, 1: wall, 2: food, 3: pacman, 4: ghost
        self.observation_space = spaces.Box(low=0, high=4,
                                            shape=grid_size, dtype=np.int32)

        self.reset()

    def reset(self):
        # Create empty grid
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Add walls randomly
        for _ in range(2):
            x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
            self.grid[x, y] = 1

        # Add food
        self.food_positions = []
        for _ in range(3):
            while True:
                x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 2
                    self.food_positions.append((x, y))
                    break

        # Place Pacman
        while True:
            x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
            if self.grid[x, y] == 0:
                self.pacman_pos = (x, y)
                self.grid[x, y] = 3
                break

        # Place ghosts
        self.ghost_positions = []
        for _ in range(self.num_ghosts):
            while True:
                x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 4
                    self.ghost_positions.append((x, y))
                    break

        self.done = False
        return self.grid.copy()

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, {}

        # Remove Pacman from old position
        x, y = self.pacman_pos
        self.grid[x, y] = 0

        # Compute new position
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = x + dx, y + dy

        # Check boundaries and walls
        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and self.grid[nx, ny] != 1:
            self.pacman_pos = (nx, ny)

        # Update Pacman on grid
        x, y = self.pacman_pos
        reward = 0

        # Eat food
        if self.grid[x, y] == 2:
            reward += 10
            self.food_positions.remove((x, y))

        self.grid[x, y] = 3

        # Move ghosts randomly
        new_ghost_positions = []
        for gx, gy in self.ghost_positions:
            self.grid[gx, gy] = 0
            gx2, gy2 = gx + random.choice([-1, 0, 1]), gy + random.choice([-1, 0, 1])
            gx2 = max(0, min(self.grid_size[0]-1, gx2))
            gy2 = max(0, min(self.grid_size[1]-1, gy2))
            if self.grid[gx2, gy2] == 0:
                gx, gy = gx2, gy2
            self.grid[gx, gy] = 4
            new_ghost_positions.append((gx, gy))
        self.ghost_positions = new_ghost_positions

        # Check collisions
        if self.pacman_pos in self.ghost_positions:
            reward -= 10
            self.done = True

        # Check win
        if not self.food_positions:
            self.done = True
            reward += 50

        return self.grid.copy(), reward, self.done, {}

    def render(self, mode='human'):
        print(self.grid)

    def close(self):
        pass
