from core.models import Route, Solution




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
