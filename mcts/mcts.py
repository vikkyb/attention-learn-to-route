from mcts.mcts_node import MCTS_node
import numpy as np

class MCTS_TSP():
    def __init__(self, graph, start_node, n_expansions:int=1, n_rollouts:int=1, 
    eval_selection:str="best", eval_rollout:str="best"):
        self.graph = graph
        self.tour = [start_node]
        self.available_nodes = []
        for i in range(len(graph)):
            if i != start_node: self.available_nodes.append(i)
        
        self.n_expansions = n_expansions
        self.n_rollouts = n_rollouts
        self.eval_selection = eval_selection
        self.eval_rollout = eval_rollout

        self.best_seen_length = len(self.graph)
        self.best_seen_tour = None

        self.root = MCTS_node(self, None, start_node, self.available_nodes, self.tour)

        self.prior_probabilities = np.ones(len(graph))

    def mcts_decide(self):
        best_outcome_since_move = len(self.graph)

        for exp in range(self.n_expansions):
            selected_node = self.root.select(best_outcome_since_move)
            selected_node = selected_node.expand()

            if selected_node in self.root._children:
                # Update prior
                selected_node.prior_p = self.prior_probabilities[selected_node.current_node]

            observed_length = selected_node.rollout(self.n_rollouts)
            if observed_length < best_outcome_since_move: best_outcome_since_move = observed_length
            
            selected_node.backpropagate(observed_length)
        
        best_next_node = self.root.get_best_child()
        # print(best_next_node)
        # print(self.best_seen_length)
        return best_next_node

    def move_to(self, node):
        self.tour.append(node)
        self.root = self.root._children[node]
        self.available_nodes.remove(node)
    
    def update_priors(self, priors):
        self.priors = priors
