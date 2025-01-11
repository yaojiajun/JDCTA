import os
import json
import random
import numpy as np

def Euclidean_fun(A, B):
    return sum((a - b)**2 for a, b in zip(A, B))**0.5

def gen_task_distribution(average_tasks_per_time_unit, max_time_span):
    return np.random.poisson(lam=average_tasks_per_time_unit, size=max_time_span)

def gen_worker_distribution(average_workers_per_time_unit, max_time_span):
    return np.random.poisson(lam=average_workers_per_time_unit, size=max_time_span)

class DataProcess:

    def task_data_fun(self, T, min_skill, max_skill, budget_range, min_pre_task_num, max_pre_task_num, min_task_deadline_span, max_task_deadline_span, DP, CP):
        K = [random.randint(min_skill, max_skill) for _ in range(T)]
        task_dict = {}
        max_time_span = 6
        average_tasks_per_time_unit = int(T / max_time_span) + 10
        tasks = gen_task_distribution(average_tasks_per_time_unit, max_time_span)

        i = cnt = 0
        for tick, task_count in enumerate(tasks):
            for _ in range(task_count):
                if cnt == T:
                    break

                LL = [random.uniform(0, 1), random.uniform(0, 1)]
                Cb = random.randint(budget_range[0], budget_range[1])
                Lt = sorted(random.sample(range(1, 4), K[i]))

                dependency = []
                if random.random() < DP:
                    dependency_count = min(random.randint(min_pre_task_num, max_pre_task_num), i)
                    dependency = sorted(random.sample(range(1 + i // 2, i + 1), dependency_count))

                arrived_time = tick + 1
                deadline = arrived_time + random.randint(min_task_deadline_span, max_task_deadline_span)

                task_dict[i + 1] = {
                    "Lt": LL,
                    "Kt": Lt,
                    "budget": Cb,
                    "Dt": dependency,
                    "Ct": [],
                    "arrived_time": arrived_time,
                    "deadline": deadline
                }

                i += 1
                cnt += 1

        conflict_dict = {}
        for t_id in task_dict:
            if random.random() < CP:
                candidate_id = random.randint(1, T)
                if candidate_id != t_id and candidate_id not in conflict_dict and candidate_id not in task_dict[t_id]["Dt"]:
                    task_dict[t_id]["Ct"].append(candidate_id)
                    task_dict[candidate_id]["Ct"].append(t_id)
                    conflict_dict[t_id] = candidate_id
                    conflict_dict[candidate_id] = t_id

        save_path = 'data/syn/worker'
        os.makedirs(save_path, exist_ok=True)
        filename = f'task{T}_{max_skill}_{min_task_deadline_span}_{DP}_{max_pre_task_num}_{CP}_U.json'
        file_path = os.path.join(save_path, filename)

        with open(file_path, 'w', encoding='utf-8') as fp_task:
            json.dump(task_dict, fp_task)

        print('Generated task dict:', len(task_dict))
        return task_dict

    def worker_data_fun(self, W, skill_quantity, skill_range, min_worker_deadline_span, max_worker_deadline_span):
        K = [random.randint(1, skill_quantity) for _ in range(W)]
        worker_dict = {}
        max_time_span = 6
        average_workers_per_time_unit = int(W / max_time_span) + 5
        workers = gen_worker_distribution(average_workers_per_time_unit, max_time_span)

        i = cnt = 0
        for tick, worker_count in enumerate(workers):
            for _ in range(worker_count):
                if cnt == W:
                    break

                worker_dict[i + 1] = {
                    "Lw": [random.uniform(0, 1), random.uniform(0, 1)],
                    "Kw": sorted(random.sample(range(1, skill_range + 1), K[i])),
                    "arrived_time": tick + 1,
                    "deadline": tick + 1 + random.randint(min_worker_deadline_span, max_worker_deadline_span),
                    "score": random.uniform(1, 100)
                }

                i += 1
                cnt += 1

        save_path = 'data/syn/worker'
        os.makedirs(save_path, exist_ok=True)
        filename = f'worker{W}_{skill_quantity}_{min_worker_deadline_span}_U.json'
        file_path = os.path.join(save_path, filename)

        with open(file_path, 'w', encoding='utf-8') as fp_worker:
            json.dump(worker_dict, fp_worker)

        print('Generated worker dict:', len(worker_dict))
        return worker_dict

# Test Code
dp = DataProcess()
task_data = dp.task_data_fun(T=2500, min_skill=1, max_skill=3, budget_range=[1, 100], min_pre_task_num=1, max_pre_task_num=3, min_task_deadline_span=6, max_task_deadline_span=6, DP=0.5, CP=0.5)
worker_data = dp.worker_data_fun(W=500, skill_quantity=3, skill_range=3, min_worker_deadline_span=6, max_worker_deadline_span=6)
