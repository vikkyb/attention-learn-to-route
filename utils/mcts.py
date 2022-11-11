import time
import torch
import math
import numpy as np
from mcts import mcts
from copy import deepcopy
from typing import NamedTuple
from utils.lexsort import torch_lexsort
from problems.tsp.state_tsp import StateTSP
from problems.tsp.problem_tsp import TSP

class MCTS():
    """
    In this class we represent a state which can be updated by taking actions. 
    We can get the possible actions and check whether the state is the terminal state.
    """
    def __init__(self):
        self.currentPlayer = 1

    def getCurrentPlayer(self):
        # should be implemented because it is used in mcts but we never switch player in takeAction()
        return self.currentPlayer


    def getPossibleActions(self):
        possibleActions = []

        # use the function propose_expansions in attention_model.py and beam_search.py
        # (instead of computing beam_size child nodes we want to compute all the child nodes of the current node)

        return possibleActions

    def takeAction(self, action):
        state = deepcopy(self.state) # we don't want to change the original state
        newState = StateTSP.update(action) # function in state_tsp.py

        return newState

    def isTerminal(self):
        # terminal is true if we reach a leaf node so if current node doesn't have child nodes (so no possible actions)
        if self.getPossibleActions == None:
            return True
        else:
            return False

    def getReward(self):
        # reward is still a bit vague but we need to do something with the tour length
        # so adding 1 for each child node is not a good way to give rewards
        raise NotImplementedError
        

if __name__=="__main__":
    initialState = MCTS() # I don't know whether this is the correct way to initialize a state with TSP
    searcher = mcts(timeLimit=1000)
    action = searcher.search(initialState=initialState)

    print(action)
