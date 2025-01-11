import os
import json
import math
import copy
import random
from random import choice
from itertools import combinations, product
import numpy as np


def Euclidean_fun(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))


def gen_task_distribution(average_tasks_per_time_unit, max_time_span):
    average_tasks_per_time_unit = average_tasks_per_time_unit
    max_time_span = max_time_span

    return np.random.poisson(lam=average_tasks_per_time_unit, size=max_time_span)


def gen_worker_distribution(average_workers_per_time_unit, max_time_span):
    average_workers_per_time_unit = average_workers_per_time_unit
    max_time_span = max_time_span

    return np.random.poisson(lam=average_workers_per_time_unit, size=max_time_span)


class data_process(object):

    def task_data_fun(self, T, min_skill, max_skill, budget_range, min_pre_task_num, max_pre_task_num,min_task_deadline_span, max_task_deadline_span, DP,CP,skill_pro):
        """
        Construct task_dict
        Randomly generate the corresponding skill requirements Kt, location, budget,
        T: number of tasks to generate
        min_pre_task_num: minimum number of dependent tasks
        max_pre_task_num: maximum number of dependent tasks
        max_conflict_num: maximum number of conflicting tasks
        """
        task_dict = {}
        i = 0
        cnt = 0
        max_time_span = 6
        average_tasks_per_time_unit =int(T/max_time_span)+10

        tasks = gen_task_distribution(
            average_tasks_per_time_unit, max_time_span)
        while cnt != T:
            break_sign = False
            if break_sign:
                break
            for tick, task_count in enumerate(tasks):
                print(f"Tick {tick + 1}: {task_count} tasks")
                for _ in range(task_count):
                    Cb = random.randint(0, 100)

                    Lt = []
                    if random.random() <= skill_pro:
                        K = []
                        skill_num = random.randint(min_skill, max_skill)
                        while len(Lt) != skill_num:
                            Cb = random.randint(50, 100)
                            x = random.randint(1, 3)
                            if x in Lt:
                                continue
                            else:
                                Lt.append(x)
                                Lt.sort()
                    else:
                        skill_value = random.randint(1, 3)
                        Lt.append(skill_value)
                        Lt.sort()
                    if cnt == T:
                        break_sign = True
                        break
                    l1 = random.uniform(0, 1)
                    l2 = random.uniform(0, 1)
                    LL = [l1, l2]

                    # Dependency
                    dependency = []
                    den_pro = random.randint(DP * 10, DP * 10) / 10
                    if random.random() <= den_pro:
                        Cb = random.randint(0, 100)
                        dependency_count = min(random.randint(
                            min_pre_task_num, max_pre_task_num), i)
                        while dependency_count >0:
                            dependency_count=dependency_count-1
                            x = random.randint(1+i//2, i)
                            if x in dependency:
                                continue
                            else:
                                dependency.append(x)
                        dependency.sort()

                    # conflict
                    conflict = []
                    # arrived time
                    arrived_time = tick+1
                    # set deadline
                    deadline = arrived_time + \
                        random.randint(min_task_deadline_span,
                                       max_task_deadline_span)
                    # add item to task_dict
                    task_dict[i + 1] = {"Lt": LL, "Kt": Lt,
                                        "budget": Cb,
                                        "Dt": dependency,
                                        "Ct": conflict,
                                        "arrived_time": arrived_time,
                                        "deadline": deadline
                                        }
                    Lt = []
                    i += 1
                    cnt += 1
                if break_sign:
                    break
        # add conflict
        conflict_dict = {}
        for t_id in task_dict:
            con_pro = random.randint(1, CP * 10) / 10
            if random.random() <= con_pro:
                candidate_id = random.randint(1, T)
                if candidate_id == t_id:
                    continue
                if candidate_id in conflict_dict:
                    continue
                if candidate_id in task_dict[t_id]["Dt"]:
                    continue
                if t_id in task_dict[candidate_id]["Dt"]:
                    continue
                task_dict[t_id]["Ct"].append(candidate_id)
                task_dict[candidate_id]["Ct"].append(t_id)
                conflict_dict[t_id] = candidate_id
                conflict_dict[candidate_id] = t_id
        period=min_task_deadline_span

        save_path = 'data/syn/test'
        # os.makedirs(save_path, exist_ok=True)
        filename = f'task{T}_{max_skill}_{period}_{DP}_{max_pre_task_num}_{CP}_U.json'

        file_path = os.path.join(save_path, filename)

        with open(file_path, 'w', encoding='utf-8') as fp_task:
            json.dump(task_dict, fp_task)
        print('generated task dict:', len(task_dict.keys()))

        return task_dict

    def worker_data_fun(self, W, skill_quantity, skill_range, min_worker_deadline_span, max_worker_deadline_span, skill_pro):
        # set default skill_quantity
        worker_dict = {}
        i = 0
        cnt = 0
        max_time_span = 6
        average_workers_per_time_unit = int(W/max_time_span)+5

        workers = gen_worker_distribution(
            average_workers_per_time_unit, max_time_span)
        while cnt != W:

            break_sign = False
            if break_sign:
                break
            for tick, worker_count in enumerate(workers):
                print(f"Tick {tick + 1}: {worker_count} workers")
                for _ in range(worker_count):
                    # build worker skill set
                    Kt = []
                    skill_pro = random.randint(1, skill_pro * 10) / 10
                    if random.random() <= skill_pro:
                        skill_num = random.randint(1, skill_quantity)
                        while len(Kt) != skill_num:
                            x = random.randint(1, 3)
                            if x in Kt:
                                continue
                            else:
                                Kt.append(x)
                                Kt.sort()
                    else:
                        skill_value = random.randint(1, 3)
                        Kt.append(skill_value)
                        Kt.sort()
                    if cnt == W:
                        break_sign = True
                        break
                    worker_dict[i+1] = {}
                    # location
                    worker_dict[i+1]['Lw'] = [
                        random.uniform(0, 1), random.uniform(0, 1)]

                    worker_dict[i+1]['Kw'] = Kt
                    worker_dict[i+1]['arrived_time'] = tick+1
                    worker_dict[i+1]['deadline'] = worker_dict[i+1]['arrived_time'] + \
                        random.randint(min_worker_deadline_span,
                                       max_worker_deadline_span)
                    worker_dict[i+1]['score'] = random.uniform(1, 100)
                    i += 1
                    cnt += 1
        period=min_worker_deadline_span

        save_path = 'data/syn/test'
        # os.makedirs(save_path, exist_ok=True)
        filename = f'worker{W}_{skill_quantity}_{period}_U.json'
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w', encoding='utf-8') as fp_worker:
            json.dump(worker_dict, fp_worker)
        print('generated  worker dict', len(worker_dict.keys()))
        return worker_dict


# test code
dp = data_process()
aa = dp.task_data_fun(T=3000, min_skill=1, max_skill=3, budget_range=[
                 1, 100], min_pre_task_num=1, max_pre_task_num=3,
                 min_task_deadline_span=6, max_task_deadline_span=6, DP=0.5, CP=0.5, skill_pro=0.5)
bb = dp.worker_data_fun(500, 1, 3, 6, 6, skill_pro=0.5)