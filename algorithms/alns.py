import math
import random
from tqdm import tqdm
import copy
from core.models import Solution
from algorithms.destruction import shaw_destroy
from algorithms.repair import regret_k_repair
from algorithms.local_search import two_opt_star

def accept_solution(current_solution, new_solution, temp):
    """
    模拟退火接受准则
    :param current_solution: 当前解
    :param new_solution: 新解
    :param temp: 当前温度
    :return: 接受的解
    """
    if new_solution.total_cost < current_solution.total_cost:
        return copy.deepcopy(new_solution)
    else:
        # 以一定概率接受劣解
        cost_diff = new_solution.total_cost - current_solution.total_cost
        accept_prob = math.exp(-cost_diff / temp)
        if random.random() < accept_prob:
            return copy.deepcopy(new_solution)
        else:
            return copy.deepcopy(current_solution)


def alns_with_shaw_regret(initial_solution, capacity, iterations=500):
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)

    # 初始化温度参数
    initial_temp = 1000
    cooling_rate = 0.99

    # 算子权重
    destroy_weights = {'random': 1, 'shaw': 1}
    repair_weights = {'greedy': 1, 'regret2': 1, 'regret3': 1}

    for it in tqdm(range(iterations)):
        current_temp = initial_temp * (cooling_rate ** it)

        # 破坏阶段
        #if random.random() < 0.5:
        #    destroyed, removed = random_destroy(current_solution, remove_ratio=0.2)
        #else:
        num_remove = int(0.2 * sum(len(r.nodes) - 2 for r in current_solution.routes))

        destroyed, removed = shaw_destroy(current_solution, num_remove,
                                              current_solution.dist_matrix,
                                              current_solution.nodes)

        # 修复阶段
        #if random.random() < 0.5:
            #repaired = greedy_repair(destroyed, removed, capacity)
        #else:
        repaired = regret_k_repair(destroyed, removed, current_solution.nodes,
                                       current_solution.dist_matrix, capacity, k=2)

        # 局部搜索
        repaired = two_opt_star(repaired, capacity)
        repaired.total_cost = sum(r.cost for r in repaired.routes)

        # 接受新解
        current_solution = accept_solution(current_solution, repaired, current_temp)

        # 更新最佳解
        if current_solution.total_cost < best_solution.total_cost:
            best_solution = copy.deepcopy(current_solution)
            print(f"Iter {it}: New best (Cost={best_solution.total_cost:.1f}, "
                  f"Vehicles={len(best_solution.routes)})")

    return best_solution

