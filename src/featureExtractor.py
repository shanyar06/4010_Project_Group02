#feature extractors for Approximate Q-Learning.
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        return a util.Counter mapping feature_name -> value
        """
        raise NotImplementedError


class SimpleExtractor(FeatureExtractor):
    """
    features for (state, action), computed on the following state:
      - bias
      - closest food pellet
      - eats food: true/false
      - number of ghosts 1 step away from current state
    """
    def getFeatures(self, state, action):
        feats = util.Counter()

        #following state after taking this action
        successor = state.generateSuccessor(0, action)
        pacPos = successor.getPacmanPosition()
        food = successor.getFood()
        walls = successor.getWalls()
        ghosts = successor.getGhostPositions()

        #bias term
        feats["bias"] = 1.0

        #ghosts 1 step away
        ghost_count = 0
        for g in ghosts:
            if util.manhattanDistance(pacPos, g) <= 1:
                ghost_count += 1
        feats["num of ghosts 1 step away"] = float(ghost_count)

        #if in danger, ignore food features
        if ghost_count > 0:
            feats.divideAll(10.0)
            return feats

        #closest food pellet distance
        foodList = food.asList()
        if len(foodList) > 0:
            closest = min(util.manhattanDistance(pacPos, f) for f in foodList)
            feats["closest food"] = closest / float(walls.width + walls.height)
        else:
            feats["closest food"] = 0.0

        prevFoodList = state.getFood().asList()
        feats["eats-food"] = 1.0 if pacPos in prevFoodList else 0.0

        #stability
        feats.divideAll(10.0)
        return feats
