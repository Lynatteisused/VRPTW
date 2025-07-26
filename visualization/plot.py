import matplotlib.pyplot as plt
#graph
def draw_routes(solution):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    for route in solution.routes:
        x = [solution.nodes[n].x for n in route.nodes]
        y = [solution.nodes[n].y for n in route.nodes]
        plt.plot(x, y, marker='o')
    plt.title(f"车辆路径图（共 {len(solution.routes)} 辆车）")
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid()
    plt.show()

def draw_time_window_distribution(solution):
    import matplotlib.pyplot as plt
    ready = [n.ready for n in solution.nodes[1:]]
    due = [n.due for n in solution.nodes[1:]]
    plt.figure(figsize=(10, 4))
    plt.hist(ready, bins=30, alpha=0.5, label='Ready Time')
    plt.hist(due, bins=30, alpha=0.5, label='Due Time')
    plt.title('时间窗分布')
    plt.legend()
    plt.show()


def plot_solution(solution, title="VRPTW Solution"):
    """
    Visualize the vehicle routing solution with enhanced features.

    Args:
        solution: Solution object containing routes and nodes
        title: Title for the plot (default: "VRPTW Solution")
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # Plot all nodes
    all_x = [node.x for node in solution.nodes[1:]]  # Customers
    all_y = [node.y for node in solution.nodes[1:]]
    depot = solution.nodes[0]

    plt.scatter(all_x, all_y, c='blue', marker='o', label='Customers', alpha=0.7)
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=150, label='Depot', edgecolors='black')

    # Plot routes with different colors
    colors = plt.cm.tab10.colors  # Use a colormap for distinct route colors
    for i, route in enumerate(solution.routes):
        x_coords = [solution.nodes[n].x for n in route.nodes]
        y_coords = [solution.nodes[n].y for n in route.nodes]

        # Draw the route path
        plt.plot(x_coords, y_coords, '--',
                 color=colors[i % len(colors)],
                 alpha=0.6,
                 linewidth=2,
                 label=f'Route {i + 1}')

        # Mark the direction of travel
        for j in range(len(x_coords) - 1):
            dx = x_coords[j + 1] - x_coords[j]
            dy = y_coords[j + 1] - y_coords[j]
            plt.arrow(x_coords[j], y_coords[j],
                      dx * 0.9, dy * 0.9,  # Scale to avoid over-reaching
                      shape='full',
                      lw=0,
                      length_includes_head=True,
                      head_width=2,
                      color=colors[i % len(colors)])

    plt.title(f"{title}\nVehicles: {len(solution.routes)} | Total Distance: {solution.total_cost:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()















def plot_solution1(sol, title="VRPTW Solution"):
    """可视化路径"""
    plt.figure(figsize=(10, 8))

    # 绘制所有节点
    x = [node.x for node in sol.nodes]
    y = [node.y for node in sol.nodes]
    plt.scatter(x, y, c='red', marker='o', label='Customers')
    plt.scatter(sol.nodes[0].x, sol.nodes[0].y, c='green', marker='s', s=100, label='Depot')

    # 绘制路径
    colors = plt.cm.tab10.colors
    route_start = 0
    for i in range(1, len(sol.path)):
        if sol.path[i].name == 0:  # 仓库节点
            x_coords = [node.x for node in sol.path[route_start:i + 1]]
            y_coords = [node.y for node in sol.path[route_start:i + 1]]
            color = colors[len(plt.gca().lines) % len(colors)]
            plt.plot(x_coords, y_coords, '--', color=color, alpha=0.6)
            route_start = i

    plt.title(f"{title} (Cost: {sol.cost:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


def plot_convergence(costs, title="Cost Convergence"):
    """绘制成本收敛曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(costs, 'b-', linewidth=1)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    plt.grid()
    plt.show()
