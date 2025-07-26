import numpy as np

def calculate_route_cost(route_nodes, dist_matrix):
    cost = 0
    for i in range(len(route_nodes) - 1):
        cost += dist_matrix[route_nodes[i]][route_nodes[i + 1]]
    return cost


def calculate_route_time(route_nodes, nodes, dist_matrix, time_matrix=None):
    """
    Calculate time-related metrics for a route with full time window propagation
    Returns: (total_time, time_windows_violated, waiting_time)
    """
    if time_matrix is None:
        time_matrix = dist_matrix  # Fallback if no time matrix provided

    current_time = 0
    prev_node = route_nodes[0]  # Start at depot
    time_windows_violated = 0
    total_waiting = 0

    for node_id in route_nodes[1:-1]:  # Skip depot at start/end
        node = nodes[node_id]

        # Calculate arrival and service times
        arrival = current_time + time_matrix[prev_node][node_id]
        waiting = max(0, node.ready - arrival)
        start = max(arrival, node.ready)
        departure = start + node.service

        # Check time window violation
        if start > node.due:
            time_windows_violated += 1

        total_waiting += waiting
        current_time = departure
        prev_node = node_id

    # Return to depot
    current_time += time_matrix[prev_node][route_nodes[-1]]

    return {
        'total_time': current_time,
        'violations': time_windows_violated,
        'waiting_time': total_waiting,
        'feasible': time_windows_violated == 0
    }



def compute_time_matrix(nodes, dist_matrix, service_times=None, speed=1.0):
    """
    Compute time matrix considering travel time and service times
    Args:
        nodes: List of Node objects
        dist_matrix: Distance matrix (distance between each node pair)
        service_times: Optional list of service times for each node
        speed: Vehicle speed (distance units per time unit)
    Returns:
        time_matrix: Matrix of minimum travel times between nodes including service times
    """
    num_nodes = len(nodes)
    time_matrix = np.zeros((num_nodes, num_nodes))

    # If no service times provided, use default from nodes or 0
    if service_times is None:
        service_times = [node.service if hasattr(node, 'service') else 0 for node in nodes]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                time_matrix[i][j] = 0
            else:
                # Travel time = distance / speed + service time at origin
                time_matrix[i][j] = (dist_matrix[i][j] / speed) + service_times[i]

    return time_matrix



def update_route_schedule(route, nodes, dist_matrix, time_matrix=None):
    """
    Update the time schedule for an entire route after modifications
    Args:
        route: Route object to update
        nodes: List of all node objects
        dist_matrix: Distance matrix
        time_matrix: Optional precomputed time matrix
    Returns:
        Updated route with correct time calculations
    """
    if time_matrix is None:
        time_matrix = dist_matrix  # Fallback to distance matrix if no time matrix

    current_time = 0
    total_cost = 0
    prev_node = route.nodes[0]  # Start at depot

    # Reset route properties
    route.load = 0
    route.time = 0
    route.feasible = True

    for i in range(1, len(route.nodes)):
        current_node = route.nodes[i]

        # Skip depot-to-depot segments
        if prev_node == 0 and current_node == 0:
            continue

        # Update load (for customer nodes only)
        if current_node != 0:
            route.load += nodes[current_node].demand

        # Calculate arrival and departure times
        travel_time = time_matrix[prev_node][current_node]
        arrival_time = current_time + travel_time
        node_data = nodes[current_node]

        if current_node != 0:  # For customer nodes
            start_time = max(arrival_time, node_data.ready)
            if start_time > node_data.due:
                route.feasible = False
            departure_time = start_time + node_data.service
        else:  # For depot nodes
            departure_time = arrival_time  # No service time at depot

        # Update cumulative values
        route.time = departure_time
        total_cost += travel_time
        current_time = departure_time
        prev_node = current_node

    route.cost = total_cost
    return route



def is_feasible_route(route_nodes, nodes, dist_matrix, capacity):
    load = 0
    current_time = 0

    for i in range(1, len(route_nodes) - 1):
        node = nodes[route_nodes[i]]
        load += node.demand
        if load > capacity:
            return False

        arrival = current_time + dist_matrix[route_nodes[i - 1]][route_nodes[i]]
        start = max(arrival, node.ready)
        if start > node.due:
            return False

        # 检查是否影响后续节点
        if i < len(route_nodes) - 2:
            next_node = nodes[route_nodes[i + 1]]
            next_arrival = start + node.service + dist_matrix[route_nodes[i]][route_nodes[i + 1]]
            if next_arrival > next_node.due:
                return False

        current_time = start + node.service

    return True

