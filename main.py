import math
from core.models import Solution
from core.models import Node
from core.distance import calculate_distance_matrix
from algorithms.initialization import solomon_i1_insertion
from algorithms.alns import alns_with_shaw_regret
from visualization.plot import draw_routes
from core.utils import is_feasible_route


def read_data():
    sol = Solution()

    filename = "data/homberger_200_customer_instances/C1_2_3.TXT"  #
    with open(filename, 'r') as f:
        for line in f.readlines()[9:]:  # 跳过前9行说明
            data = list(filter(None, line.strip().split()))
            node = Node()
            node.name = int(data[0])
            node.x = int(data[1])
            node.y = int(data[2])
            node.demand = int(data[3])
            node.ready = int(data[4])
            node.due = int(data[5])
            node.service = int(data[6])
            sol.nodes.append(node)

    sol.dist_matrix = calculate_distance_matrix(sol.nodes)
    return sol

def calculate_distance_matrix(nodes):
    n = len(nodes)
    matrix = [[0.0] * n for v_ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = nodes[i].x - nodes[j].x
            dy = nodes[i].y - nodes[j].y
            matrix[i][j] = math.hypot(dx, dy)
    return matrix


if __name__ == "__main__":
    CAPACITY = 200
    sol = read_data()

    # 诊断信息
    total_demand = sum(n.demand for n in sol.nodes[1:])
    min_vehicles = math.ceil(total_demand / CAPACITY)
    print(f"诊断信息：总需求={total_demand} 理论最少车辆={min_vehicles}")

    #sol = kmeans_initial_solution(sol, CAPACITY, 50)
    #print(f"初始解：{len(sol.routes)}辆车 总成本={sol.total_cost:.2f}")
    solomon_i1_insertion(sol, CAPACITY)

    best = alns_with_shaw_regret(sol, CAPACITY, 500)

    # 验证最终解
    print("\n最终解验证：")
    print(f"- 车辆数：{len(best.routes)}")
    print(f"- 总载重：{sum(r.load for r in best.routes)}/{total_demand}")
    print(f"- 平均载重：{sum(r.load for r in best.routes) / len(best.routes):.1f}/{CAPACITY}")

    # 检查时间窗
    for i, r in enumerate(best.routes):
        if not is_feasible_route(r.nodes, best.nodes, best.dist_matrix, CAPACITY):
            print(f"警告：路径{i}违反约束！")

    #draw_routes(best)
    plot_solution(best)