import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ghostAgents
import multiAgents
import pacman as pacmanFile # assuming pacman.py is in the same folder or properly importable
import layout as layoutFile

class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}


class PacmanEnv(gym.Env):
    def __init__(self, layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
        super(PacmanEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.grid_size = (11, 20)
        self.observation_space = spaces.Box(low=0, high=5, shape=self.grid_size, dtype=np.int8)
        self.agent = pacman
        self.rules = pacmanFile.ClassicGameRules(timeout)
        self.reward = 0 
        self.ghosts = [ghostAgents.RandomGhost(i+1) for i in range(2)]

        # Initialize game (does NOT run it - game does not start until first step in test loop)
        self.game = self.rules.newGame(layout, self.agent, self.ghosts, display, False, catchExceptions)
        self.game.display.initialize(self.game.state.data)

        self.done = False

    def reset(self, layout, pacman, ghosts, display, numGames, record,  catchExceptions=False, timeout=30):
        # Reset the game
        self.game = self.rules.newGame(layout, pacman, 
                                        ghosts, display, False,
                                        catchExceptions)
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        # Convert game state to a numeric array
        # 0 = empty, 1 = wall, 2 = food, 3 = pacman, 4 = ghost
        grid = np.zeros(self.grid_size, dtype=np.int8)
        
        state = self.game.state

        walls = state.getWalls()
        food = state.getFood()
        pacmanPos = state.getPacmanPosition()
        ghostPos = state.getGhostPositions()

        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    grid[y, x] = 1


        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    grid[y, x] = 2

        grid[int(pacmanPos[1]), int(pacmanPos[0])] = 3
        
        for gpos in ghostPos:
            grid[int(gpos[1]), int(gpos[0])] = 4

        return grid

    
    def step(self):

        # Generate action for Pacman agent - greedy policy
        state = self.game.state
        action = self.agent.getAction(state)

        # Update state 
        state = self.game.state
        next_state = state.generateSuccessor(0, action)
        self.game.state = next_state

        # Update display based on action 
        if hasattr(self.game, 'display') and self.game.display is not None:
            self.game.display.update(self.game.state.data)

        # Generate action for each ghost, update state and display
        for i, ghost in enumerate(self.ghosts):
            self.done = self.game.state.isWin() or self.game.state.isLose()
            if(self.done):
                break

            g_action = ghost.getAction(self.game.state)
 
            self.game.state = self.game.state.generateSuccessor(i+1, g_action)
            if hasattr(self.game, 'display') and self.game.display is not None:
                self.game.display.update(self.game.state.data)


        # Compute reward and done
        prevReward = self.reward
        self.reward = self.game.state.getScore()

        self.done = self.game.state.isWin() or self.game.state.isLose()

        obs = self._get_observation()

        return obs, prevReward, self.done, {}
    
    def render(self, final_reward = None, episode = None, num_steps = None):
        # rendering a human readable episode summary after done is true.

        # making sure that the game exists
        if not hasattr(self, "game") or self.game is None:
            print("Nothing to render")
            return
        
        state = self.game.state
        if state.isWin():
            result = "Won"
        if state.isLose():
            result = "Lost"
        
        # Displaying
        print("-----------------------")
        print("Pac-Man Episode Summary")
        print("-----------------------")

        if episode is not None:
            print(f"Episode: {episode}")
        
        if final_reward is not None:
            print(f"Final reward is {final_reward}")

        if num_steps is not None:
            print(f"It took the agent {num_steps} steps")
        
        print(f"Overall Pac-Man outcome: {result}")
    def close(self):
        pass
