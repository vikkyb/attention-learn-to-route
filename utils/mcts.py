import numpy as np
import math
from copy import deepcopy

def scaled_sigmoid(new_length, best_length) -> float:
    """
    Calculate scaled Sigmoid function based on the new length and best length

    Args:
    new_length : float
        represents the length score 
    
    best_length : float
        represents the best score seen until then

    Returns:
    result : float
    """
    extra_scaling_parameter = 5
    return 1 / (1 + math.exp(extra_scaling_parameter * (new_length - best_length)))


def evaluate_tour(graph, tour) -> float:
    """
    Evaluate a tour on a graph by calculating the length of the tour

    Args:
    graph : 2D numpy array
        describes the TSP graph by giving (x, y) coordinates per node
    
    tour : list of ints
        gives the indexes of nodes that were visited in the order they were visited 

    Returns:
    tour_length : float
        gives a total traversed length of the tour 
    """
    tour_length = 0
    for i, node in enumerate(tour):
        if i != len(tour) - 1:
            # Simple Euclidean distance
            tour_length += np.linalg.norm(graph[node] - graph[tour[i + 1]])
    return tour_length


class NodeTSP():
    def __init__(self, graph, current_node, available_nodes, tour, parent, prior_p, is_root=False):
        """
        Create an MCTS node
        
        Attributes:
        graph : 2D numpy array
            describes the TSP graph by giving (x, y) coordinates per node

        current_node : int
            gives the index of the current node in the graph

        available_nodes : list of ints
            a list that contains the indexes of nodes that can be visited
        
        tour : list of ints
            gives the indexes of nodes that were visited in the order they were visited
        
        parent : NodeTSP class or None (if root)
            If root, parent is None. Else, parent points to the parent

        prior_p : float
            prior probability of selecting the node, is only applicable to root-level children
            if parent is not root, prior_p = 1
        
        is_root : bool
            True if this node is the root, else False
        """
        self.graph = graph
        self.current_node = current_node
        self.available_nodes = available_nodes 
        self.tour = tour
        self._children = {}
        self.parent = parent

        self.is_root = is_root
        self.prior_p = prior_p
        self.tour_lengths = []
       

    def select(self, best_score):
        """
        Select a child node (if available) and recursively call the select function
        until leaf or terminal node 

        Returns:
        best_child.select() : function call
            call the select() on the best child

        OR

        best_child : NodeTSP
            return the best child that is either leaf or terminal
        """
        if self.is_terminal():
            # Terminal state
            return self

        elif self.is_leaf():
            """
            Please ignore this commented section, it functions identical
            to the self.is_terminal() section
            """
            # Leaf node, expand and get all available nodes
            # self.expand() # TODO: DIT MOET NIET
            # Select a random child to perform rollout on
            # random_child_key = np.random.choice(list(self._children.keys()))
            # return self._children[random_child_key]
            return self

        else:
            # Recursively select best child node until terminal or leaf, there is probably a more convenient way to do this
            values = {}
            for child in self._children:
                if child not in self.tour:
                    values[self._children[child].current_node] = self._children[child].get_score(best_score)
            best_child_key = max(values, key=values.get)
            return self._children[best_child_key].select(best_score)


    def expand(self):
        """
        Perform the expansion phase of MCTS by creating nodes for all possible children.       
        """
        for n in self.available_nodes:
            # Note that this is skipped if the node is terminal
            new_available_nodes = list(set(self.available_nodes) - set([n]))
            tour_copy = deepcopy(self.tour)
            tour_copy.append(n)
            # TODO: Create option for using GNN prior
            prior = 1 # This should be settable when the GNN is implemented
            self._children[n] = NodeTSP(self.graph, n, new_available_nodes, tour_copy, self, prior)

    
    def rollout(self, n_rollouts, eval_mode="mean"):
        """
        Perform rollout n_rollout times and yield back either the average or best result

        Args:
        n_rollouts : int
            the number of rollouts to be performed

        eval_mode : ["mean", "best"]
            indicates whether the best result or the average result should be returned

        Returns:
        summarized_rollout_outcomes : float
            either the mean or average length as result of this node
        """
        assert eval_mode in ["mean", "best"], "eval_mode is not properly set, use best or mean"

        if self.is_terminal():
            # Don't perform rollout if terminal, just return the observed length of the generated tour
            return evaluate_tour(self.graph, self.tour)

        else:
            # Perform rollout
            rollout_outcomes = []
            for r in range(n_rollouts):
                rollout_outcomes.append(self.one_rollout())
                if eval_mode == "mean": return np.mean(rollout_outcomes)
                elif eval_mode == "best": return np.min(rollout_outcomes)
        
    
    def one_rollout(self):
        """
        Not exactly performing a rollout, but the effect is the same.
        We simulate a rollout by shuffling the remaining available nodes, which returns a random tour, and adding the starting node at the end
        
        Returns:
        new_tour_length : float
            the length of the rollout tour
        """
        copy_available_nodes = deepcopy(self.available_nodes)
        np.random.shuffle(copy_available_nodes)
        new_tour = self.tour + copy_available_nodes + [0] # DO NOT FORGET TO FINISH ON THE NODE WE STARTED ON
        new_tour_length = evaluate_tour(self.graph, new_tour)
        return new_tour_length      


    def backprop(self, rollout_outcome):
        """
        Backpropagate the length to the parent nodes

        Args:
        rollout_outcome : float
            rollout_outcome refers to the (best or average) length that should be propagated returned from rollout
        """
        self.update_recursive(rollout_outcome)
    

    def update_recursive(self, update_value):
        """
        Recursively perform updates over the parents' parents given the update value (rollout outcome passed from backprop())

        Args:
        update_value : float
            identical to rollout_outcome in backprop, (best or average) length that should be propagated returned from rollout
        """
        self.tour_lengths.append(update_value)
        if self.parent != None:
            self.parent.update_recursive(update_value)


    def get_score(self, best_score, eval_mode="mean", c_exploration=1, no_explore=False) -> float:
        """
        TODO: Rewrite this function to favor legibility
        The score is given by two elements: the q value and the exploration element. Exploration is given by UCT equation without winrate.
        As winrate cannot be used for TSP, we use a scaled sigmoid function that we pass the difference between the best tour length and the
        observed tour length, resulting in a q value. This ensures that better outcomes are selected more often.

        The sum of the scaled sigmoid function (q) and the exploration element are multiplied with the prior probablity.

        Args:
        best_score : float
            best observed score to compare against

        eval_mode : ["mean", "best"]
            indicates whether scores are calculated based on average or best outcomes
        
        c_exploration : float
            exploration parameter used in the UCT value

        no_explore : bool
            only set to true in the final evaluation to decide where to move next, disregards the exploration part of UCT
        
        """
        assert eval_mode in ["mean", "best"], "eval_mode is not properly set, use best or mean"
        if len(self.tour_lengths) != 0:
            if eval_mode == "mean":
                score = np.mean(self.tour_lengths)
            elif eval_mode == "best":
                score = np.min(self.tour_lengths)
            q = scaled_sigmoid(score, best_score)
            if self.parent != None:
                exploration = c_exploration * np.sqrt(np.log(len(self.parent.tour_lengths) + 1) / len(self.tour_lengths))
            else:
                exploration = c_exploration

            if no_explore: 
                return q
            else:
                return self.prior_p * (q + exploration)

        else:
            exploration = c_exploration
            if no_explore: exploration = 0
                # This should actually never occur but you never know
            return self.prior_p * exploration


    def is_leaf(self) -> bool:
        """
        Returns:
        _is_leaf : bool
            True if self is leaf, else False
        """
        if len(self._children) == 0:
            return True
        else:
            return False


    def is_terminal(self) -> bool:
        """
        Returns:
        _is_terminal : bool
            True if self is terminal, else False
        """
        if len(self.graph) == len(self.tour):
            return True
        else:
            return False
    

    def make_root(self):
        """
        Transform the self node into the root and decouple the parent
        """
        self.is_root = True
        self.parent = None

    
    def remove_node_from_available(self, node):
        """
        When you move to a node, remove it from any available nodes

        Args:
        node : NodeTSP
            node to be removed
        """
        if node in self._children:
            del self._children[node]
        for c in self._children:
            self._children[c].remove_node_from_available(node)


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

        self.root = NodeTSP(self.graph, self.current_node, self.available_nodes, self.tour, None, 1, True)


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
        for run in range(self.n_runs):
            selected_node = self.root.select(self.best_score)
            selected_node.expand()
            outcome = selected_node.rollout(self.n_rollout, self.eval_mode)
            selected_node.backprop(outcome)
            self.best_score = np.min([outcome, self.best_score])

        values = {}
        for child in self.root._children:
            values[self.root._children[child].current_node] = self.root._children[child].get_score(self.best_score, no_explore=True)
        best_child_key = max(values, key=values.get)
        return best_child_key