import numpy as np
import math 

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
    extra_scaling_parameter = 3
    return 1 / (1 + math.exp(extra_scaling_parameter * (best_length - new_length)))


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
    ordered_points = graph[tour]
    tour_length = np.sum(np.sqrt(np.sum(np.diff(ordered_points, axis=0)**2, 1)))
    # tour_length = 0 
    # for i, node in enumerate(tour):
    #     if i < len(tour) - 1:
    #         # Simple Euclidean distance
    #         tour_length += np.linalg.norm(graph[node] - graph[tour[i + 1]])
    # print(tour_length)
    return tour_length