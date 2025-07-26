import random
import copy
import numpy as np
from collections import defaultdict
from core.models import Route
from core.utils import calculate_route_cost,calculate_route_time


capacity=200
#Solomon I1 initiate
def solomon_i1_insertion(sol, capacity):
    """基于 Solomon I1 插入启发式构造初始解"""
    customers = sol.nodes[1:]  # 去掉仓库节点（编号为0）
    depot = sol.nodes[0]
    unassigned = customers.copy()
    routes = []

    while unassigned:
        route = Route()
        route.nodes = [0]  # 从仓库出发
        route.load = 0
        route.time = 0
        route.cost = 0

        # 1. 选取最早 ready_time 的客户作为路径起点
        seed = min(unassigned, key=lambda c: c.ready)
        unassigned.remove(seed)
        route.nodes.append(seed.name)
        route.load += seed.demand
        arrival = sol.dist_matrix[0][seed.name]
        begin_service = max(arrival, seed.ready)
        route.time = begin_service + seed.service
        route.cost += sol.dist_matrix[0][seed.name]
        route.cost += sol.dist_matrix[seed.name][0]  # 暂时回到仓库

        inserted = True
        while inserted and unassigned:
            best_increase = float('inf')
            best_position = -1
            best_customer = None

            for customer in unassigned:
                for i in range(1, len(route.nodes)):  # 从1开始，0是仓库
                    prev = route.nodes[i - 1]
                    next_ = route.nodes[i] if i < len(route.nodes) else 0
                    increase = (sol.dist_matrix[prev][customer.name] +
                                sol.dist_matrix[customer.name][next_] -
                                sol.dist_matrix[prev][next_])

                    if route.load + customer.demand > capacity:
                        continue

                    arrival_time = route.time + sol.dist_matrix[route.nodes[-1]][customer.name]
                    start_time = max(arrival_time, customer.ready)
                    if start_time > customer.due:
                        continue

                    if increase < best_increase:
                        best_increase = increase
                        best_position = i
                        best_customer = customer

            if best_customer:
                unassigned.remove(best_customer)
                route.nodes.insert(best_position, best_customer.name)
                route.load += best_customer.demand
                route.time = max(route.time + sol.dist_matrix[route.nodes[-2]][best_customer.name], best_customer.ready) + best_customer.service
                route.cost += best_increase
            else:
                inserted = False

        route.nodes.append(0)  # 回到仓库
        route.cost += sol.dist_matrix[route.nodes[-2]][0]
        routes.append(route)

    # 汇总最终解
    sol.routes = routes
    sol.total_cost = sum(r.cost for r in routes)
    return sol

#destroy
def shaw_destroy(solution, num_remove, dist_matrix, nodes, time_matrix=None, adaptive_weights=True):
    """Enhanced similarity-based destruction operator"""
    routes = solution.routes

    # Collect all customer nodes (excluding depot)
    customers = []
    customer_route_map = {}
    for route in routes:
        route_customers = route.nodes[1:-1]
        customers.extend(route_customers)
        for c in route_customers:
            customer_route_map[c] = route

    if len(customers) <= num_remove:
        return solution, customers

    # Adaptive selection of seed customer
    seed = select_seed_customer(customers, nodes, dist_matrix)
    removed = [seed]
    candidates = set(c for c in customers if c != seed)

    # Adaptive similarity weights
    if adaptive_weights:
        weights = calculate_adaptive_weights(solution, nodes)
    else:
        weights = {'distance': 0.4, 'time': 0.4, 'demand': 0.2}

    while len(removed) < num_remove and candidates:
        # Calculate similarity scores
        similarities = []
        current_route = customer_route_map[seed]

        for c in candidates:
            # Enhanced similarity components
            dist_sim = dist_matrix[seed][c] / np.percentile(dist_matrix, 95)

            # Time window similarity (both ready and due times)
            time_sim = (abs(nodes[seed].ready - nodes[c].ready) +
                        abs(nodes[seed].due - nodes[c].due)) / 2880  # Max 2*1440

            # Demand similarity (relative to capacity)
            demand_sim = abs(nodes[seed].demand - nodes[c].demand) / capacity

            # Route context similarity
            route_sim = 0 if customer_route_map[c] == current_route else 0.2

            # Combined similarity
            similarity = (weights['distance'] * dist_sim +
                          weights['time'] * time_sim +
                          weights['demand'] * demand_sim +
                          route_sim)

            # Add controlled randomness
            similarity += random.uniform(0, weights.get('randomness', 0.2))

            similarities.append((c, similarity))

        # Select next customer to remove
        similarities.sort(key=lambda x: x[1])
        next_customer = similarities[0][0]
        removed.append(next_customer)
        candidates.remove(next_customer)
        seed = next_customer  # Chain to next customer

    # Remove customers from routes
    new_routes = []
    for route in routes:
        new_nodes = [n for n in route.nodes if n not in removed]
        if len(new_nodes) > 2:  # Keep non-empty routes
            new_route = Route()
            new_route.nodes = new_nodes
            new_route.load = sum(nodes[n].demand for n in new_nodes[1:-1])
            new_route.time = calculate_route_time(new_nodes, nodes, dist_matrix, time_matrix)
            new_route.cost = calculate_route_cost(new_nodes, dist_matrix)
            new_routes.append(new_route)

    destroyed = copy.deepcopy(solution)
    destroyed.routes = new_routes
    return destroyed, removed


def select_seed_customer(customers, nodes, dist_matrix):
    """Select seed customer based on centrality and time window tightness"""
    # Score each customer
    scores = []
    for c in customers:
        # Centrality score (average distance to others)
        centrality = sum(dist_matrix[c][other] for other in customers) / len(customers)

        # Time window tightness
        tw_width = nodes[c].due - nodes[c].ready
        tightness = 1 / (1 + tw_width)

        # Combined score (higher is better)
        scores.append((c, tightness * 0.7 + (1 / centrality) * 0.3))

    # Select top 5 most "interesting" candidates randomly
    scores.sort(key=lambda x: -x[1])
    return random.choice(scores[:5])[0]


def calculate_adaptive_weights(solution, nodes):
    """Dynamically adjust similarity weights based on solution characteristics"""
    # 1. Calculate time window utilization (using total_time from time data)
    total_time = 0
    for route in solution.routes:
        if not hasattr(route, 'time_data'):  # Ensure time data exists
            route.time_data = calculate_route_time(route.nodes, nodes, solution.dist_matrix)
        total_time += route.time_data['total_time']

    tw_utilization = total_time / (len(solution.routes) * 1440)  # 1440 = max minutes in day

    # 2. Analyze demand distribution
    demands = [n.demand for n in nodes[1:]]  # Skip depot
    avg_demand = sum(demands) / len(demands) if demands else 0
    demand_variation = np.std(demands) if len(demands) > 1 else 0

    # 3. Adjust weights dynamically
    weights = {
        'distance': 0.4,
        'time': 0.3 + 0.2 * min(tw_utilization, 1.0),  # Cap at 1.0
        'demand': max(0.1, 0.3 - 0.1 * (demand_variation / avg_demand)) if avg_demand > 0 else 0.2,
        'randomness': 0.1
    }

    # Normalize weights (excluding randomness)
    total = sum(v for k, v in weights.items() if k != 'randomness')
    for k in weights:
        if k != 'randomness':
            weights[k] /= total

    return weights



def random_destroy(solution, remove_ratio=0.2):
    new_solution = copy.deepcopy(solution)
    num_customers = sum(len(r.nodes) - 2 for r in new_solution.routes)
    num_remove = int(num_customers * remove_ratio)
    removed = set()
    while len(removed) < num_remove:
        r = random.choice(new_solution.routes)
        if len(r.nodes) > 2:
            i = random.randint(1, len(r.nodes) - 2)
            removed.add(r.nodes.pop(i))
    return new_solution, list(removed)

