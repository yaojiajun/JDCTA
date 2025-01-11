import copy
import random
from utils.score_tool import dependency_check, satisfy_check, conflict_check, Euclidean_fun, satisfaction

def price_score(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = sum(Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) for i in worker_list)
        return task_dict[t]['budget'] - dis * v
    return 0

def temp_satisfaction(task_dict, worker_dict, t, move_cost, cooperation_workers, alpha, worker_v, max_s, max_p):
    if not cooperation_workers or max_p == 0:
        return 0
    total_score = sum(worker_dict[w]['score'] for w in cooperation_workers)
    profit_w = price_score(task_dict, worker_dict, t, move_cost, cooperation_workers)
    return alpha * (total_score / len(cooperation_workers)) / max_s + (1 - alpha) * (profit_w / max_p)

def update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, success_to_assign_task, delete_to_assign_task):
    for t_id in task_dict.keys():
        if task_dict[t_id]['deadline'] < current_time or tuple(task_dict[t_id]['Dt']) in delete_to_assign_task:
            delete_to_assign_task.add(t_id)

    for t_id in task_dict.keys():
        if (task_dict[t_id]['arrived_time'] <= current_time <= task_dict[t_id]['deadline'] and
                t_id not in success_to_assign_task and tuple(task_dict[t_id]['Ct']) not in success_to_assign_task and
                tuple(task_dict[t_id]['Dt']) not in delete_to_assign_task and t_id not in delete_to_assign_task):
            arrived_tasks.add(t_id)

    for w_id in worker_dict.keys():
        if worker_dict[w_id]['arrived_time'] <= current_time <= worker_dict[w_id]['deadline']:
            arrived_workers.add(w_id)

    return arrived_tasks, arrived_workers

def greedy_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_s, max_p, worker_v, reachable_dis):
    full_assign = 0
    task_assign_condition = {i: -1 for i in task_dict.keys()}
    best_assign = {
        i: {
            'list': [],
            'group': {k: 0 for k in task_dict[i]['Kt']},
            'satisfaction': 0,
            'assigned': False
        }
        for i in task_dict.keys()
    }

    success_to_assign_task = []
    delete_to_assign_task = set()

    for current_time in range(1, max_time + 1):
        arrived_tasks = set()
        arrived_workers = set()
        assigned_workers = set()
        success_to_assign_worker = set()
        arrived_tasks, arrived_workers = update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, success_to_assign_task, delete_to_assign_task)

        disrupted_tasks = list(arrived_tasks)
        random.shuffle(disrupted_tasks)

        skill_group = {i: {k: [] for k in task_dict[i]['Kt']} for i in task_dict.keys()}

        for i in disrupted_tasks:
            task = task_dict[i]

            if not dependency_check(i, task_assign_condition, task_dict):
                continue

            if conflict_check(i, task_assign_condition, task_dict):
                continue

            candidate_worker = [
                w_id for w_id in arrived_workers
                if satisfy_check(task, worker_dict[w_id], task['deadline'], worker_dict[w_id]['deadline'], current_time, worker_v, move_cost, reachable_dis)
            ]

            if not candidate_worker:
                continue

            for k in task['Kt']:
                for j in candidate_worker:
                    if k in worker_dict[j]['Kw']:
                        skill_group[i][k].append(j)

            worker_list = []
            success_assign_flag = False

            for r in skill_group[i].keys():
                skill_list = list(set(skill_group[i][r]).difference(assigned_workers))
                if skill_list:
                    greedy_best_pick = -1
                    greedy_best_sat = -1
                    for current_worker in skill_list:
                        worker_list.append(current_worker)
                        current_sat = temp_satisfaction(task_dict, worker_dict, i, move_cost, worker_list, alpha, worker_v, max_s, max_p)
                        if current_sat > greedy_best_sat:
                            greedy_best_sat = current_sat
                            greedy_best_pick = current_worker
                        worker_list.pop()
                    worker_list.append(greedy_best_pick)
                    assigned_workers.add(greedy_best_pick)

            if len(worker_list) == len(task['Kt']):
                success_assign_flag = True
            else:
                for k in worker_list:
                    assigned_workers.discard(k)

            if success_assign_flag:
                task_assign_condition[i] = current_time
                full_assign += 1
                success_to_assign_task.append(i)
                best_assign[i]['list'] = worker_list

                for w in worker_list:
                    best_assign[i]['group'][worker_dict[w]['Kw'][0]] = w
                    success_to_assign_worker.add(w)
                    assigned_workers.update(success_to_assign_worker)

                best_assign[i]['satisfaction'] = satisfaction(task_dict, worker_dict, i, move_cost, worker_list, alpha, worker_v, max_s, max_p, current_time)
                best_assign[i]['assigned'] = True

                arrived_tasks -= set(success_to_assign_task)
                arrived_workers -= set(success_to_assign_worker)

    total_sat = sum(best_assign[i]['satisfaction'] for i in best_assign.keys())
    return task_assign_condition, best_assign, full_assign, total_sat
