class Node:
    def __init__(self):
        self.name = 0
        self.x = 0
        self.y = 0
        self.demand = 0
        self.ready = 0
        self.due = 0
        self.service = 0

class Route:
    def __init__(self):
        self.nodes = []
        self.load = 0
        self.time = 0
        self.cost = 0

class Solution:
    def __init__(self):
        self.routes = []
        self.nodes = []
        self.dist_matrix = []
        self.total_cost = 0
        self.feasible = True