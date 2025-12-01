# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action)
                      for action in legal]
        scored = [(self.evaluationFunction(state), action)
                  for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

# Tabular Q-Learning agent that will update Q using the previous state/action and the current state
class TabularQAgent(Agent):
    # alpha is the rate at which Q-values will update
    # gamma is the discount factor
    # epsilon is the exploration rate which indicates how likely we are to take a random action
    def __init__(self, alpha = 0.1, gamma = 0.99, epsilon = 0.1):
        # super().__init__(index=0) is called implicitly
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        # Making a Q-table dictionary where some (state, action) pair maps to a value. Starts as an empty dictionary.
        self.Q_table = {}

        # A Q-learning update requires us to keep track of prev_state, prev_action, prev_reward, and curr_state. Storing the first 3 all in variables and initializing them none/0.0.
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0.0
        
    # We need to be able to represent the state into a format that the Q-Learning algorithm can store in its Q-table. This is done in the representState function.
        # This must be hashable and compact
    def representState(self, state):
        pacman_position = state.getPacmanPosition() # Pacman's current position coordinates
        ghosts_position = tuple(sorted(state.getGhostPositions())) # Tuple of the ghosts's position coordinates (sorted so that ghost positions dont matter)
        food_count = state.getNumFood() # Stores the amount of remaining food pellets
        return (pacman_position, ghosts_position, food_count)

    # Choosing an action to take based on the epsilon greedy policy and Q-Learning
    def getAction(self, state):
        state_representation = self.representState(state)

        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # choosing the exploration route with probability equal to epsilon
        # otherwise exploiting by choosing the action with the largest Q-value
        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            # For every action (N, E, S, W), looking up and storing the Q-value for the state-action pair (state_representation, action) in the Q-table (self.Q_table).
            # If it does not exist in the table, then returning a default value of 0.0 (which is used for unknown state value pairs)
            q_values = [self.Q_table.get((state_representation, action), 0.0) for action in legal] 

            # storing the actions that have the max q_value
            max_q_value = max(q_values)

            best_actions = []
            for action, q_value in zip(legal, q_values):
                if q_value == max_q_value:
                    best_actions.append(action)
            
            # choosing one of the best actions randomly for our action
            action = random.choice(best_actions) 

        # Current state-action pair is needed when updating the Q-table
        self.prev_state = state_representation
        self.prev_action = action

        return action

    def updateQ(self, state, reward):
        # using the previous state and previous action to update the Q-value based on the reward and the next state's best action
        state_representation = self.representState(state)

        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Finding out what is the maximum Q-value for the next state
        # If there are legal actions, loop through all of them, and use the Q-table to get their Q-values
        if legal:
            # starting with -infinity as the max_q_next_state, then looping through
            max_q_next_state = -float('inf')

            # for every action, getting the q_value for the state-action pair (using 0.0 if not found), and updating the max if our found value is greater
            for action in legal:
                # Getting the q_value for the state_action_pair
                q_value_state_action = self.Q_table.get((state_representation, action), 0.0)

                if q_value_state_action > max_q_next_state:
                    max_q_next_state = q_value_state_action
        else:
            max_q_next_state = 0.0

        # Storing the q_value for the previous state-action pair (defaults to 0.0 if not found)
        prev_q_value = self.Q_table.get((self.prev_state, self.prev_action), 0.0)
        # Computing the new q_value using the q_learning formula
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_next_state - prev_q_value)

        # Updating the Q-table entry for the previous state-action pair with the new Q-value
        self.Q_table[(self.prev_state, self.prev_action)] = new_q_value

def scoreEvaluation(state):
    return state.getScore()
