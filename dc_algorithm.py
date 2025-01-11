import copy
import random
from itertools import combinations
from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, Euclidean_fun, satisfaction

def price_score(task_dict, worker_dict, t, v, worker_list):
    if not worker_list:
        return 0
    dis = sum(Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) for i in worker_list)
    return task_dict[t]['budget'] - dis * v

def temp_satisfaction(task_dict, worker_dict, t, move_cost, cooperation_workers, alpha, worker_v, max_s, max_p):
    if not cooperation_workers or max_p == 0:
        return 0
    total_score = sum(worker_dict[w]['score'] for w in cooperation_workers)
    profit_w = price_score(task_dict, worker_dict, t, move_cost, cooperation_workers)
    return alpha * (total_score / len(cooperation_workers)) / max_s + (1 - alpha) * (profit_w / max_p)

def get_free_task(task_dict, arrived_tasks, best_assign):
    free_task = set()
    for i in arrived_tasks:
        if all(best_assign[depend_id]['assigned'] for depend_id in task_dict[i]['Dt']) and \
           all(not best_assign[conflict_id]['assigned'] for conflict_id in task_dict[i]['Ct']):
            free_task.add(i)
    return free_task

def update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, success_to_assign_task, delete_to_assign_task):
    for t_id in task_dict.keys():
        if task_dict[t_id]['deadline'] < current_time or tuple(task_dict[t_id]['Dt']) in delete_to_assign_task:
            delete_to_assign_task.add(t_id)

    for t_id in task_dict.keys():
        if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
           t_id not in success_to_assign_task and tuple(task_dict[t_id]['Ct']) not in success_to_assign_task and \
           tuple(task_dict[t_id]['Dt']) not in delete_to_assign_task and t_id not in delete_to_assign_task:
            arrived_tasks.add(t_id)
    for w_id in worker_dict.keys():
        if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
            arrived_workers.add(w_id)
    return arrived_tasks, arrived_workers

def divide_and_conquer(task_dict, worker_dict, task_ids, skill_group, move_cost, alpha, max_s, max_p, worker_v, current_time, reachable_dis, arrived_workers, assigned_workers):
    if len(task_ids) <= 2:
        assignments = []
        for task_id in task_ids:
            task = task_dict[task_id]
            required_skills = task['Kt']
            candidate_workers = [w_id for w_id in arrived_workers if satisfy_check(task, worker_dict[w_id], task['deadline'], worker_dict[w_id]['deadline'], current_time, worker_v, move_cost, reachable_dis)]
            for k in range(len(task['Kt'])):
                for j in candidate_workers:
                    if task['Kt'][k] in worker_dict[j]['Kw']:
                        skill_group[task_id][task['Kt'][k]].append(j)

            worker_list = []
            for r in skill_group[task_id].keys():
                skill_list = list(set(skill_group[task_id][r]).difference(assigned_workers))
                if skill_list:
                    best_worker = min(skill_list, key=lambda w: Euclidean_fun(task_dict[task_id]['Lt'], worker_dict[w]['Lw']))
                    worker_list.append(best_worker)

            if len(worker_list) == len(required_skills):
                assigned_workers.update(worker_list)
                assignments.append((task_id, worker_list))

        return assignments

    mid = len(task_ids) // 2
    left_tasks = task_ids[:mid]
    right_tasks = task_ids[mid:]

    left_assignment = divide_and_conquer(task_dict, worker_dict, left_tasks, skill_group, move_cost, alpha, max_s, max_p, worker_v, current_time, reachable_dis, arrived_workers, assigned_workers)
    right_assignment = divide_and_conquer(task_dict, worker_dict, right_tasks, skill_group, move_cost, alpha, max_s, max_p, worker_v, current_time, reachable_dis, arrived_workers, assigned_workers)

    return left_assignment + right_assignment

def g_dc_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_s, max_p, worker_v, reachable_dis):
    full_assign = 0
    task_assign_condition = {i: -1 for i in task_dict.keys()}
    best_assign = {i: {'list': [], 'group': {k: 0 for k in task_dict[i]['Kt']}, 'satisfaction': 0, 'assigned': False} for i in task_dict.keys()}

    success_to_assign_task = []
    delete_to_assign_task = set()
    for current_time in range(1, max_time + 1):
        arrived_tasks = set()
        arrived_workers = set()
        assigned_workers = set()
        success_to_assign_worker = set()

        arrived_tasks, arrived_workers = update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, success_to_assign_task, delete_to_assign_task)
        arrived_tasks = get_free_task(task_dict, arrived_tasks, best_assign)

        skill_group = {i: {k: [] for k in task_dict[i]['Kt']} for i in task_dict.keys()}

        task_worker_pairs = divide_and_conquer(task_dict, worker_dict, list(arrived_tasks), skill_group, move_cost, alpha, max_s, max_p, worker_v, current_time, reachable_dis, list(arrived_workers), assigned_workers)

        for task_id, worker_list in task_worker_pairs:
            if task_id in success_to_assign_task:
                continue

            if worker_list:
                task_assign_condition[task_id] = current_time
                full_assign += 1
                success_to_assign_task.append(task_id)
                best_assign[task_id]['list'] = worker_list
                for w in worker_list:
                    best_assign[task_id]['group'][worker_dict[w]['Kw'][0]] = w
                    success_to_assign_worker.add(w)
                    assigned_workers.update(success_to_assign_worker)
                cur_task_satisfaction = satisfaction(task_dict, worker_dict, task_id, move_cost, worker_list, alpha, worker_v, max_s, max_p, current_time)
                best_assign[task_id]['satisfaction'] = cur_task_satisfaction
                best_assign[task_id]['assigned'] = True

                arrived_tasks -= set(success_to_assign_task)
                arrived_workers -= set(success_to_assign_worker)

    total_sat = sum(best_assign[i]['satisfaction'] for i in best_assign.keys())
    return task_assign_condition, best_assign, full_assign, total_sat
