from mcts.mcts_node import MCTS_node
import numpy as np

class MCTS_TSP():
    def __init__(self, graph, start_node, n_runs=100, n_rollout=1, gnn_model=None, eval_mode="mean"):
        """
        The main MCTS class that is used to model the TSP problem
        TODO: Finish documentation
        """
        self.graph = graph
        self.current_node = start_node
        self.tour = [start_node]
        self.available_nodes = []
        for i in range(len(graph)): 
            if i != start_node: self.available_nodes.append(i)
        self.n_runs = n_runs
        self.n_rollout = n_rollout
        self.gnn_model = gnn_model
        self.eval_mode = eval_mode

        self.best_score = len(graph)

        self.root = MCTS_node(self.graph, self.current_node, self.available_nodes, self.tour, None, 1, True)


    def move_to(self, node):
        """
        In the tree, move to the supplied node

        Args:
        node : NodeTSP
            move to the selected node and update it as the root 
        """
        self.root = self.root._children[node]
        self.tour.append(node)
        self.root.make_root()
        self.available_nodes.remove(node)
        self.root.remove_node_from_available(self)


    def update_priors(self, priors):
        """
        Update the prior probabilities of the root's children.

        Args:
        priors : 1d np_array
            array of size n_nodes containing prior probabilities for those nodes
        """
        for c in self.root._children:
            self.root._children[c].prior_p = priors[c]


    def mcts_decide(self):
        """
        Perform an MCTS run to find the best action given the current node.
        
        TODO: Finish documentation
        """
        self.best_outcome = len(self.graph) # reset each time
        for run in range(self.n_runs):
            selected_node = self.root.select(self.best_score, self.eval_mode)
            selected_node.expand()
            outcome = selected_node.rollout(self.n_rollout, self.eval_mode)
            selected_node.update_recursive(outcome)
            self.best_score = np.min([outcome, self.best_score])
            self.best_outcome = np.min([self.best_outcome, outcome])

        values = {}
        for child in self.root._children:
            values[self.root._children[child].current_node] = np.min(self.root._children[child].tour_lengths)
        best_child_key = min(values, key=values.get)
        return best_child_key