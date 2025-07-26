import random
import copy
import math
from core.models import Route
from core.utils import update_route_schedule,calculate_route_cost
from core.utils import compute_time_matrix

capacity=200

def regret_k_repair(solution, removed_nodes, nodes, dist_matrix, capacity, k=3, time_matrix=None, vehicle_penalty=1000):
    """Enhanced Regret-k repair with proper time propagation and vehicle count control"""

    # Precompute earliest/latest arrival times for efficiency
    if time_matrix is None:
        time_matrix = compute_time_matrix(nodes, dist_matrix)

    while removed_nodes:
        regrets = []
        new_route_candidates = []

        for node_id in removed_nodes:
            node = nodes[node_id]
            best_insertions = []

            # 1. Evaluate insertions in existing routes
            for route_idx, route in enumerate(solution.routes):
                if route.load + node.demand > capacity:
                    continue

                # Evaluate all possible insertion positions
                for insert_pos in range(1, len(route.nodes)):
                    # Check time window feasibility with full propagation
                    feasible, cost = evaluate_insertion(
                        route, node_id, insert_pos,
                        nodes, dist_matrix, time_matrix)

                    if feasible:
                        best_insertions.append((cost, route_idx, insert_pos))

            # 2. Evaluate new route creation (with penalty)
            new_route_cost = dist_matrix[0][node_id] + dist_matrix[node_id][0]
            best_insertions.append((
                new_route_cost + vehicle_penalty * (1 + len(solution.routes) / 200),
                -1,  # Special index for new route
                1  # Insert position in new route
            ))

            if not best_insertions:
                continue  # Shouldn't happen since we have new route option

            # Sort insertions by cost
            best_insertions.sort(key=lambda x: x[0])

            # Calculate enhanced regret value
            if len(best_insertions) >= k:
                regret = sum(best_insertions[i][0] - best_insertions[0][0]
                             for i in range(1, k))
            else:
                regret = sum(best_insertions[i][0] - best_insertions[0][0]
                             for i in range(1, len(best_insertions)))

            # Add diversity factor
            diversity = calculate_diversity(node_id, best_insertions[0][1], solution)
            regret *= diversity

            regrets.append((node_id, regret, best_insertions))

        if not regrets:
            break

        # Select node with maximum regret
        regrets.sort(key=lambda x: -x[1])
        selected_node, _, best_insertions = regrets[0]

        # Perform the best insertion
        min_cost, route_idx, insert_pos = best_insertions[0]

        if route_idx == -1:  # New route case
            new_route = Route()
            new_route.nodes = [0, selected_node, 0]
            new_route.load = nodes[selected_node].demand
            new_route.cost = min_cost - vehicle_penalty
            solution.routes.append(new_route)
        else:  # Existing route
            route = solution.routes[route_idx]
            route.nodes.insert(insert_pos, selected_node)
            route.load += nodes[selected_node].demand

            # Update route times and cost
            update_route_schedule(route, nodes, dist_matrix, time_matrix)
            route.cost = calculate_route_cost(route.nodes, dist_matrix)

        removed_nodes.remove(selected_node)

    # Post-optimization to reduce vehicles
    solution = reduce_vehicles(solution, nodes, dist_matrix, capacity, time_matrix)

    return solution

def evaluate_insertion(route, node_id, insert_pos, nodes, dist_matrix, time_matrix):
    """Comprehensive insertion evaluation with time propagation"""
    # Make a temporary copy for evaluation
    temp_route = copy.deepcopy(route)
    temp_route.nodes.insert(insert_pos, node_id)

    # Check time feasibility through entire route
    current_time = 0
    prev_node = 0
    total_cost = 0

    for i, node in enumerate(temp_route.nodes[1:], 1):
        # Calculate arrival time
        arrival = current_time + dist_matrix[prev_node][node]
        start = max(arrival, nodes[node].ready)

        # Check time window
        if start > nodes[node].due:
            return False, float('inf')

        # Calculate departure time
        departure = start + nodes[node].service
        current_time = departure
        prev_node = node

        # Calculate cost component for this segment
        if i == insert_pos:
            # Special handling for inserted node
            prev = temp_route.nodes[i - 1]
            next_ = temp_route.nodes[i + 1] if i + 1 < len(temp_route.nodes) else 0
            delta = (dist_matrix[prev][node] +
                     dist_matrix[node][next_] -
                     dist_matrix[prev][next_])
            total_cost += delta

    return True, total_cost


def calculate_diversity(node_id, target_route_idx, solution):
    """
    Calculate how "different" a candidate insertion is from existing route
    Args:
        node_id: ID of node being inserted
        target_route_idx: Index of target route (-1 for new route)
        solution: Current solution object
        capacity: Global vehicle capacity (passed from main ALNS)
    Returns:
        diversity score (higher = more diverse/more valuable insertion)
    """
    if target_route_idx == -1:  # New route case
        return 1.5  # Bonus for considering new routes

    target_route = solution.routes[target_route_idx]
    node = solution.nodes[node_id]

    # 1. Spatial diversity (distance to route centroid)
    route_nodes = [solution.nodes[n] for n in target_route.nodes[1:-1]]
    if not route_nodes:  # Handle empty routes
        return 1.0

    centroid_x = sum(n.x for n in route_nodes) / len(route_nodes)
    centroid_y = sum(n.y for n in route_nodes) / len(route_nodes)
    distance_to_centroid = math.hypot(node.x - centroid_x, node.y - centroid_y)

    # 2. Temporal diversity (time window alignment)
    avg_ready = sum(n.ready for n in route_nodes) / len(route_nodes)
    avg_due = sum(n.due for n in route_nodes) / len(route_nodes)
    time_diff = abs(node.ready - avg_ready) + abs(node.due - avg_due)

    # 3. Load diversity
    avg_demand = sum(n.demand for n in route_nodes) / len(route_nodes)
    demand_diff = abs(node.demand - avg_demand)

    # Normalization factors
    max_dist = math.hypot(
        max(n.x for n in solution.nodes) - min(n.x for n in solution.nodes),
        max(n.y for n in solution.nodes) - min(n.y for n in solution.nodes)
    )
    normalized_dist = distance_to_centroid / max_dist if max_dist > 0 else 0

    max_time_diff = 1440  # Max possible time window difference (24 hours)
    normalized_time = time_diff / max_time_diff

    normalized_demand = demand_diff / capacity  # Now using passed capacity parameter

    # Weighted combination (adjust weights as needed)
    diversity_score = (0.5 * normalized_dist +
                       0.3 * normalized_time +
                       0.2 * normalized_demand)

    return 1.0 + diversity_score  # Ensure score > 1



def reduce_vehicles(solution, nodes, dist_matrix, capacity, time_matrix):
    """Attempt to reduce number of vehicles after repair"""
    changed = True
    while changed:
        changed = False
        # Try to move customers from small routes to others
        solution.routes.sort(key=lambda r: len(r.nodes))

        for small_route in solution.routes[:]:
            if len(small_route.nodes) <= 3:  # Only depot + 1 customer + depot
                customers = small_route.nodes[1:-1]
                for customer in customers:
                    # Try to insert into other routes
                    best_insertion = None
                    best_cost = float('inf')

                    for target_route in solution.routes:
                        if target_route == small_route:
                            continue

                        if target_route.load + nodes[customer].demand > capacity:
                            continue

                        # Evaluate all possible insertions
                        for pos in range(1, len(target_route.nodes)):
                            feasible, cost = evaluate_insertion(
                                target_route, customer, pos,
                                nodes, dist_matrix, time_matrix)

                            if feasible and cost < best_cost:
                                best_insertion = (target_route, pos, cost)
                                best_cost = cost

                    if best_insertion:
                        # Perform the move
                        target_route, pos, _ = best_insertion
                        target_route.nodes.insert(pos, customer)
                        target_route.load += nodes[customer].demand
                        update_route_schedule(target_route, nodes, dist_matrix, time_matrix)
                        target_route.cost = calculate_route_cost(target_route.nodes, dist_matrix)

                        small_route.nodes.remove(customer)
                        small_route.load -= nodes[customer].demand
                        changed = True

                # Remove empty route
                if len(small_route.nodes) <= 2:
                    solution.routes.remove(small_route)
                    changed = True
                    break

    return solution
