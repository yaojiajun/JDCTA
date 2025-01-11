from utils.score_tool import (
    dependency_check, satisfy_check, distance_check, conflict_check, Euclidean_fun, satisfaction
)
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm_max_profit(profit_matrix):
    """
    Hungarian algorithm to maximize total profit using SciPy's linear_sum_assignment.
    :param profit_matrix: NxM matrix of profits
    :return: optimal matching and total profit
    """
    profit_matrix = np.array(profit_matrix)
    max_value = profit_matrix.max()
    cost_matrix = max_value - profit_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_profit = profit_matrix[row_ind, col_ind].sum()
    matching = list(zip(row_ind, col_ind))
    return matching, total_profit


def price_score(task_dict, worker_dict, t, v, worker_list):
    """
    Calculate the profit based on distance and budget for a task.
    """
    if worker_list:
        dis = sum(Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) for i in worker_list)
        return task_dict[t]['budget'] - dis * v
    return 0


def satisfaction_score(task_dict, worker_dict, t, v, w, alpha, worker_v, max_s, max_p):
    """
    Calculate satisfaction score based on worker reputation and profit.
    """
    total_score = worker_dict[w]['score']
    profit_w = price_score(task_dict, worker_dict, t, v, [w])
    return alpha * (total_score / max_s) + (1 - alpha) * (profit_w / max_p)


def init_best_assign(task_dict):
    """
    Initialize the best assignment structure for all tasks.
    """
    best_assign = {
        i: {'list': [], 'group': {}, 'satisfaction': 0, 'assigned': False} for i in task_dict
    }
    for i in task_dict:
        for k in task_dict[i]['Kt']:
            best_assign[i]['group'][k] = 0
    return best_assign


def update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, success_tasks, delete_tasks):
    """
    Update the available tasks and workers based on current time and conditions.
    """
    delete_tasks.update(
        t_id for t_id in task_dict if
        task_dict[t_id]['deadline'] <= current_time or
        set(task_dict[t_id]['Ct']) & success_tasks or
        tuple(task_dict[t_id]['Dt']) in delete_tasks
    )

    arrived_tasks.update(
        t_id for t_id in task_dict
        if task_dict[t_id]['arrived_time'] <= current_time <= task_dict[t_id]['deadline']
        and t_id not in success_tasks
        and tuple(task_dict[t_id]['Ct']) not in success_tasks
        and tuple(task_dict[t_id]['Dt']) not in delete_tasks
    )

    arrived_workers.update(
        w_id for w_id in worker_dict
        if worker_dict[w_id]['arrived_time'] <= current_time <= worker_dict[w_id]['deadline']
    )

    return arrived_tasks, arrived_workers


def build_dependency_graph(task_dict, arrived_tasks):
    """
    Build a dependency graph for tasks based on their dependencies.
    """
    dependency_graph = {}
    for t in sorted(arrived_tasks, reverse=True):
        task = task_dict[t]
        dependencies = [dep for dep in task['Dt'] if dep in arrived_tasks]
        dependencies.append(t)
        dependency_graph[t] = sorted(dependencies)
    return dependency_graph


def get_available_tasks(task_dict, arrived_tasks, best_assign, current_time, dependency_graph, delete_tasks):
    """
    Get the list of tasks that are free to be assigned based on dependencies and conflicts.
    """
    free_tasks = []
    for task in arrived_tasks:
        related_tasks = dependency_graph.get(task, [])
        is_free = True

        for rel_task in related_tasks:
            for conflict_id in task_dict[rel_task]['Ct']:
                if best_assign[conflict_id]['assigned']:
                    is_free = False
                    break
            if not is_free:
                break

        if is_free:
            free_tasks.append(task)

    return free_tasks


def group_greedy_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_p, worker_v, reachable_dis, max_s):
    """
    Group greedy algorithm for task-worker assignment.
    """
    # Initialize performance metrics and assignment structures
    full_assign = 0
    task_assign_condition = {i: -1 for i in task_dict}
    best_assign = init_best_assign(task_dict)

    arrived_tasks, arrived_workers = set(), set()
    success_tasks, delete_tasks = set(), set()
    assigned_workers = set()

    # Iterate over each time slice
    for current_time in range(1, max_time + 1):
        arrived_tasks, arrived_workers = update_task_worker(
            task_dict, worker_dict, arrived_tasks, arrived_workers,
            current_time, success_tasks, delete_tasks
        )

        dependency_graph = build_dependency_graph(task_dict, arrived_tasks)
        available_tasks = get_available_tasks(
            task_dict, arrived_tasks, best_assign, current_time, dependency_graph, delete_tasks
        )

        # Assign workers to tasks
        for task in available_tasks:
            task_data = task_dict[task]
            candidates = [
                worker for worker in arrived_workers
                if satisfy_check(task_data, worker_dict[worker], task_data['deadline'],
                                 worker_dict[worker]['deadline'], current_time, worker_v, move_cost, reachable_dis)
            ]

            if candidates:
                # Calculate satisfaction scores and assign best candidate
                best_worker = max(
                    candidates,
                    key=lambda w: satisfaction_score(
                        task_dict, worker_dict, task, move_cost, w, alpha, worker_v, max_s, max_p
                    )
                )
                best_assign[task]['list'].append(best_worker)
                best_assign[task]['assigned'] = True
                best_assign[task]['satisfaction'] = satisfaction(
                    task_dict, worker_dict, task, move_cost, [best_worker], alpha, worker_v, max_s, max_p, current_time
                )
                success_tasks.add(task)
                assigned_workers.add(best_worker)

        # Update metrics
        full_assign += len(success_tasks)
        arrived_tasks -= success_tasks
        arrived_workers -= assigned_workers

    # Calculate total satisfaction
    total_satisfaction = sum(assign['satisfaction'] for assign in best_assign.values())
    return task_assign_condition, best_assign, full_assign, total_satisfaction
