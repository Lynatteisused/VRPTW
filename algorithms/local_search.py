import random
from core.models import Route
from core.utils import calculate_route_cost,calculate_route_time,compute_time_matrix

def two_opt_star(solution, capacity, time_matrix=None, max_iterations=100):
    """Enhanced 2-opt* with proper time window handling and vehicle reduction"""

    if time_matrix is None:
        time_matrix = compute_time_matrix(solution.nodes, solution.dist_matrix)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        best_move = None
        best_improvement = 0

        # Evaluate all possible route pairs
        for i in range(len(solution.routes)):
            for j in range(i + 1, len(solution.routes)):
                r1 = solution.routes[i]
                r2 = solution.routes[j]

                original_cost = r1.cost + r2.cost

                # Try all possible exchange points
                for a in range(1, len(r1.nodes) - 1):
                    for b in range(1, len(r2.nodes) - 1):
                        # Create new routes by swapping tails
                        new_r1 = Route()
                        new_r1.nodes = r1.nodes[:a] + r2.nodes[b:-1] + [0]

                        new_r2 = Route()
                        new_r2.nodes = r2.nodes[:b] + r1.nodes[a:-1] + [0]

                        # Skip if routes become too small
                        if len(new_r1.nodes) <= 2 or len(new_r2.nodes) <= 2:
                            continue

                        # Check capacity and time windows
                        if (is_route_feasible(new_r1, solution.nodes, capacity, time_matrix) and
                                is_route_feasible(new_r2, solution.nodes, capacity, time_matrix)):

                            # Calculate new costs
                            new_r1.cost = calculate_route_cost(new_r1.nodes, solution.dist_matrix)
                            new_r2.cost = calculate_route_cost(new_r2.nodes, solution.dist_matrix)
                            new_cost = new_r1.cost + new_r2.cost

                            improvement = original_cost - new_cost

                            # Track best improvement
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (i, j, a, b, new_r1, new_r2)

        # Apply the best move if found (outside the nested loops)
        if best_improvement > 0:
            i, j, a, b, new_r1, new_r2 = best_move
            solution.routes[i] = new_r1
            solution.routes[j] = new_r2
            solution.total_cost -= best_improvement
            improved = True

            # Try to eliminate empty routes
            solution = eliminate_empty_routes(solution)

        iteration += 1

    # Additional optimization: try to merge small routes (outside main loop)
    solution = merge_routes(solution, capacity, time_matrix)

    return solution

def is_route_feasible(route, nodes, capacity, time_matrix):
    """Check if route satisfies capacity and time windows"""
    load = 0
    current_time = 0
    prev_node = 0  # Depot

    for node_id in route.nodes[1:-1]:  # Skip depot at both ends
        node = nodes[node_id]

        # Check capacity
        load += node.demand
        if load > capacity:
            return False

        # Check time window
        arrival = current_time + time_matrix[prev_node][node_id]
        start = max(arrival, node.ready)
        if start > node.due:
            return False

        current_time = start + node.service
        prev_node = node_id

    return True


def eliminate_empty_routes(solution):
    """Remove routes with only depots"""
    solution.routes = [r for r in solution.routes if len(r.nodes) > 2]
    return solution


def merge_routes(solution, capacity, time_matrix):
    """Attempt to merge small routes"""
    changed = True
    while changed:
        changed = False
        solution.routes.sort(key=lambda r: len(r.nodes))

        for i in range(len(solution.routes)):
            r1 = solution.routes[i]
            if len(r1.nodes) > 4:  # Only try to merge small routes
                continue

            for j in range(i + 1, len(solution.routes)):
                r2 = solution.routes[j]

                # Check combined capacity
                combined_load = r1.load + r2.load
                if combined_load > capacity:
                    continue

                # Try all possible merge combinations
                for merge_pos in range(1, len(r1.nodes)):
                    new_nodes = r1.nodes[:merge_pos] + r2.nodes[1:-1] + r1.nodes[merge_pos:]

                    # Create temporary route for feasibility check
                    temp_route = Route()
                    temp_route.nodes = new_nodes

                    # Check feasibility
                    if is_route_feasible(temp_route, solution.nodes, capacity, time_matrix):
                        # Create merged route
                        merged = Route()
                        merged.nodes = new_nodes
                        merged.load = combined_load
                        merged.cost = calculate_route_cost(new_nodes, solution.dist_matrix)
                        merged.time = calculate_route_time(new_nodes, solution.nodes, solution.dist_matrix, time_matrix)

                        # Update solution
                        solution.routes[i] = merged
                        solution.routes.pop(j)
                        solution.total_cost = sum(r.cost for r in solution.routes)
                        changed = True
                        break

                if changed:
                    break
            if changed:
                break

    return solution
