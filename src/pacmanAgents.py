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


import pandas as pd
from matplotlib import pyplot as plt
from pacman import Directions
from game import Agent
import random
import game
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from featureExtractor import SimpleExtractor

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

# DQN Agent Policy 

class DQNAgent(Agent):
    def __init__(self, epsilon=0.1, learningRate = 0.01, discountFactor=0.9, batchSize=64, targetUpdateFreq=1000, numTraining=0):
        super().__init__()
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.batchSize = batchSize
        self.targetUpdateFreq = targetUpdateFreq
        self.numTraining = numTraining

        self.state_dim = 220  # 11 x 20 grid size might need to adjust based on observation space
        self.action_dim = 5 # up, down, left, right, stop 

        # Q-network and target network
        self.q_network = QNetwork(self.state_dim, self.action_dim)

        self.target_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Adam optimizer - update neural network weights 
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learningRate) 

        # replay buffer
        # store past experiences for training   
        self.replay_buffer = ReplayBuffer()
        self.steps_done = 0

        self.reward_history = []
        self.loss_history = []

        self.last_state = None  
        self.last_action = None

    
    def getAction(self, state):
        observations = self._get_observation(state)
        flat_obs = torch.FloatTensor(observations).view(-1)  # flatten (220,)
        legal = state.getLegalPacmanActions()

        if random.random() < self.epsilon:
            action = random.choice(legal)
            action_index = self._action_to_index(action)
        else:
            with torch.no_grad():
                q_values = self.q_network(flat_obs.unsqueeze(0))
                action_index = q_values.argmax().item()
                action = self._index_to_action(action_index, legal)

        # observe transition if previous state exists
        if self.last_state is not None and self.last_action is not None:
            # reward is difference in score
            reward = state.getScore() - self.last_score
            done = state.isWin() or state.isLose()
            self.observeTransition(self.last_state, self.last_action, flat_obs.numpy(), reward, done)

        self.last_state = flat_obs.numpy()
        self.last_action = action_index
        self.last_score = state.getScore()

        return action

    def _get_observation(self, state):
        grid = np.zeros((11,20), dtype=np.float32)

        walls = state.getWalls()
        food = state.getFood()
        pacmanPos = state.getPacmanPosition()
        ghostPos = state.getGhostPositions()
        capsules = state.getCapsules()

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

        for cpos in capsules:
            grid[int(cpos[1]), int(cpos[0])] = 5

        return grid

    def _action_to_index(self, action):
        action_map = {
            Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST: 2,
            Directions.WEST: 3,
            Directions.STOP: 4
        }
        return action_map[action]
    
    def _index_to_action(self, index, legal_actions):
        index_map = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.STOP
        }
        action = index_map[index]
        if action in legal_actions:
            return action
        else:
            return random.choice(legal_actions)
    
    def observeTransition(self, old_state, action, new_state, reward, done):
        self.replay_buffer.push(old_state, action, reward, new_state, done)
        self.reward_history.append(reward)
        loss = self.trainStep()
        if loss is not None:
            self.loss_history.append(loss.item())

    def trainStep(self):
        if len(self.replay_buffer) < self.batchSize:
            return  # not enough samples to train

        # sample batch
        states, actions, rewards, next_states, done = self.replay_buffer.sample(self.batchSize)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done).unsqueeze(1)

        # compute current Q values
        curr_q_values = self.q_network(states).gather(1, actions)

        # compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.discountFactor * next_q_values * (1 - done))

        # compute loss
        loss = F.mse_loss(curr_q_values, target_q_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.steps_done += 1
        if self.steps_done % self.targetUpdateFreq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss 
    
    
class ReplayBuffer: 
    def __init__(self, capacity=10000):
        # store transitions 
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # add transition to buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # sample a batch of transitions
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    # neural network for approximating Q-values
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ApproximateQAgent(Agent):
    '''
    implementation of Approximate Q-Learning Agent
    uses SimpleExtractor and util.Counter.
    '''
    def __init__(self,
                 epsilon=0.1,
                 learningRate=0.01,
                 discountFactor=0.9,
                 numTraining=0):
        super().__init__()

        self.epsilon = float(epsilon)
        self.alpha = float(learningRate)
        self.gamma = float(discountFactor)
        self.numTraining = int(numTraining)

        self.featExtractor = SimpleExtractor()
        self.weights = util.Counter()

        self.last_state = None
        self.last_action = None
        self.last_score = 0.0
        self.episodes_so_far = 0

    #q-value
    def getQValue(self, state, action):
        '''
        Q(s,a) = w · f(s,a)
        '''
        feats = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * feats[f] for f in feats)

    def computeValue(self, state):
        '''
        V(s) = max_a Q(s,a) 
        '''
        actions = state.getLegalPacmanActions()
        if not actions: return 0.0
        return max(self.getQValue(state, a) for a in actions)

    def computeAction(self, state):
        '''
        greedy policy: argmax_a Q(s,a)
        '''
        actions = state.getLegalPacmanActions()
        if not actions:
            return None

        values = [(self.getQValue(state, a), a) for a in actions]
        max_val = max(values, key=lambda x: x[0])[0]

        best_actions = [a for (v, a) in values if v == max_val]
        return random.choice(best_actions)

    def getAction(self, state):
        '''
        epsilon-greedy action selection.
        Performs update for previous transition.
        '''
        #update weights for previous step
        if self.last_state is not None and self.last_action is not None:
            reward = state.getScore() - self.last_score
            done = state.isWin() or state.isLose()
            self.update(self.last_state, self.last_action, state, reward, done)

        legal = state.getLegalPacmanActions()
        if not legal: return None

        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action = self.computeAction(state)

        #store step for next update
        self.last_state = state
        self.last_action = action
        self.last_score = state.getScore()

        return action

    #q-learning update
    def update(self, state, action, nextState, reward, terminal):
        '''
        w_i <- w_i + alpha * (r + gamma max_a' Q(s',a') - Q(s,a)) * f_i(s,a)
        '''
        feats = self.featExtractor.getFeatures(state, action)
        current_q = self.getQValue(state, action)

        if terminal: target = reward
        else:
            target = reward + self.gamma * self.computeValue(nextState)

        td_error = target - current_q

        for f in feats:
            self.weights[f] += self.alpha * td_error * feats[f]


    def final(self, state):
        '''
        final update.
        '''
        if self.last_state is not None and self.last_action is not None:
            reward = state.getScore() - self.last_score
            self.update(self.last_state, self.last_action, state, reward, terminal=True)

        self.episodes_so_far = self.episodes_so_far + 1
        self.last_state = None
        self.last_action = None
        self.last_score = 0.0

        # turn off exploration after training
        if self.episodes_so_far >= self.numTraining:
            self.epsilon = 0.0

class PPOAgent(Agent):
    def __init__(self, lr=0.01, gamma=0.99, clip_eps=0.2):
        self.torch = torch
        self.nn = nn
        self.optim = optim

        self.gamma = gamma
        self.clip_eps = clip_eps

        # Networks will be initialized lazily
        self.policy = None
        self.value = None
        self.policy_opt = None
        self.value_opt = None

        # Stores memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def get_observations(self, state):
        pac = state.getPacmanPosition()
        ghosts = state.getGhostPositions()

        features = [pac[0], pac[1]]
        for g in ghosts:
            features.append(g[0] - pac[0])
            features.append(g[1] - pac[1])

        return features

    def init_networks(self, state_dim, action_dim):
        if self.policy is not None:
            return
        
        # Neural Networks
        nn = self.nn
        torch = self.torch
        optim = self.optim

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=0.01)
        self.value_opt = optim.Adam(self.value.parameters(), lr=0.01)

    def select_action(self, state_vec, legal_actions):
        torch = self.torch

        probs = self.policy(torch.tensor(state_vec, dtype=torch.float32))
        dist = torch.distributions.Categorical(probs)

        action_idx = dist.sample()
        logprob = dist.log_prob(action_idx)

        # store memory
        self.states.append(state_vec)
        self.actions.append(action_idx)
        self.logprobs.append(logprob)

        return legal_actions[action_idx.item()]

    def update(self):
        if not self.states:
            return

        torch = self.torch

        # Convert to tensors
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.stack(self.actions)
        old_logprobs = torch.stack(self.logprobs)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Compute advantages
        values = self.value(states).squeeze()
        adv = returns - values.detach()

        # PPO Policy loss
        probs = self.policy(states)
        dist = torch.distributions.Categorical(probs)
        new_logprobs = dist.log_prob(actions)

        ratio = torch.exp(new_logprobs - old_logprobs)

        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()

        # Optimize
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        # Reset memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        s = self.get_observations(state)

        # Lazy init network
        self.init_networks(state_dim=len(s), action_dim=len(legal))

        # Pick action
        action = self.select_action(s, legal)
        return action

    