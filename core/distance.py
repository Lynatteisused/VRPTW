import math
import numpy as np


def calculate_distance_matrix(nodes):
    n = len(nodes)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = nodes[i].x - nodes[j].x
            dy = nodes[i].y - nodes[j].y
            matrix[i][j] = math.hypot(dx, dy)
    return matrix


def compute_time_matrix(nodes, dist_matrix, service_times=None, speed=1.0):
    num_nodes = len(nodes)
    time_matrix = np.zeros((num_nodes, num_nodes))

    if service_times is None:
        service_times = [node.service if hasattr(node, 'service') else 0 for node in nodes]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                time_matrix[i][j] = 0
            else:
                time_matrix[i][j] = (dist_matrix[i][j] / speed) + service_times[i]
    return time_matrix