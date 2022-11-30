import numpy as np
from copy import deepcopy
from mcts.mcts_utils import evaluate_tour, scaled_sigmoid

class MCTS_node():
    def __init__(self, mcts_manager, parent,
        current_node:int, available_nodes:list, tour:list, prior_p:float=1.0):

        self.mcts_manager = mcts_manager
        self.parent = parent
        self.current_node = current_node
        self.available_nodes = deepcopy(available_nodes)
        self.unexplored_nodes = deepcopy(available_nodes)
        self.tour = tour
        self.prior_p = prior_p

        self.length_visits = []
        
        self._children = {}
        assert len(self.tour) + len(self.available_nodes) == len(self.mcts_manager.graph), "Tour + available nodes does not add up to graph len"


    def select(self, best_seen_score):
        if self.is_terminal() or self.is_leaf():
            return self
        
        else:
            values = {}
            for child in self._children:
                values[child] = self._children[child].get_score(best_seen_score)
            best_child_key = max(values, key=values.get)
            best_child = self._children[best_child_key].select(best_seen_score)
            return best_child

    def expand(self):
        if self.is_leaf():
            np.random.shuffle(self.unexplored_nodes)
            new_node = self.unexplored_nodes.pop(0)
            new_node_available_nodes = deepcopy(self.available_nodes)
            if new_node in new_node_available_nodes: new_node_available_nodes.remove(new_node)
            new_node_tour = self.tour + [new_node]
            self._children[new_node] = MCTS_node(self.mcts_manager, self, new_node, new_node_available_nodes, new_node_tour)
            return self._children[new_node]
        else:
            return self

    def rollout(self, n_rollouts):
        if self.is_terminal():
            rollout_result = evaluate_tour(self.mcts_manager.graph, self.tour)
            return rollout_result
        else:
            rollout_results = np.zeros(n_rollouts)
            current_tour_length = evaluate_tour(self.mcts_manager.graph, self.tour)
            copy_available_nodes = deepcopy(self.available_nodes)
            for i in range(n_rollouts):
                np.random.shuffle(copy_available_nodes)
                new_tour_length = current_tour_length + evaluate_tour(self.mcts_manager.graph, [self.current_node] + copy_available_nodes + [self.tour[0]])
                rollout_results[i] = new_tour_length
                
                if new_tour_length < self.mcts_manager.best_seen_length:
                    self.mcts_manager.best_seen_length = new_tour_length
                    self.mcts_manager.best_seen_tour = self.tour + copy_available_nodes + [self.tour[0]]
            
            if self.mcts_manager.eval_rollout == "best":
                return np.min(rollout_results)
            else:
                return np.mean(rollout_results)

    def backpropagate(self, value_to_propagate):
        self.length_visits.append(value_to_propagate)
        if self.parent != None:
            self.parent.backpropagate(value_to_propagate)

    def get_score(self, best_seen_length, exploration_constant=np.sqrt(2)):
        if self.mcts_manager.eval_selection == "best":
            node_length = np.min(self.length_visits)
        else:
            node_length = np.mean(self.length_visits)

        q_value = scaled_sigmoid(node_length, best_seen_length)
        # In the exploration equation, both parts of division might need a +1
        exploration = exploration_constant * np.sqrt(np.log(len(self.parent.length_visits)) / len(self.length_visits))
        return self.prior_p + exploration + q_value

    def get_best_child(self):
        values = {}
        for child in self._children:
            if self.mcts_manager.eval_selection == "best":
                child_length = np.min(self._children[child].length_visits)
            else:
                child_length = np.mean(self._children[child].length_visits)
            # print(self._children[child].current_node, child_length)
            values[child] = child_length
        best_child_key = min(values, key=values.get)
        return best_child_key


    def is_leaf(self) -> bool:
        if len(self.unexplored_nodes) > 0:
            return True
        else:
            return False
    
    def is_terminal(self) -> bool:
        """
        Multiple checks should any fail
        """
        if len(self.available_nodes) == 0:
            return True
        elif len(self.tour) == len(self.mcts_manager.graph):
            return True
        else:
            return False

    def make_root(self):
        self.parent = None

if __name__ == "__main__":
    print("This file contains mcts_node")
