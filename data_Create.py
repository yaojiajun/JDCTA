import random
import json
from random import choice

class DataProcess:

    def task_date_fun(self, T, min_task, max_task, budget_range):
        K = [random.randint(min_task, max_task) for _ in range(T)]
        task_dict = {}
        for i in range(T):
            LL = [random.uniform(0, 1), random.uniform(0, 1)]
            Cb = random.randint(budget_range[0], budget_range[1])
            Lt = sorted(random.sample(range(1, 11), K[i]))
            task_dict[i + 1] = {"Lt": LL, "Kt": Lt, "Cbt": Cb}
        return task_dict

    def worker_date_fun(self, W, min_worker, max_worker):
        K = [random.randint(min_worker, max_worker) for _ in range(W)]
        worker_dict = {}
        for i in range(W):
            LL = [random.uniform(0, 1), random.uniform(0, 1)]
            Cb = random.randint(5, 10)
            Lw = sorted(random.sample(range(1, 11), K[i]))
            worker_dict[i + 1] = {"Lw": LL, "Kw": Lw, "Cbw": Cb}
        return worker_dict

    def real_task_fun(self):
        with open('Real-data//task_800.txt', 'r', encoding='utf-8') as f1:
            task_dict = {}
            counttask = 0
            for line in f1.readlines():
                counttask += 1
                b = list(map(float, line.strip().split(' ')))
                task_dict[counttask] = {
                    'Lt': b,
                    'Kt': sorted(random.sample(range(1, 4), 3)),
                    'budget': random.randint(5, 10)
                }
        return task_dict

    def real_worker_fun(self):
        with open('Real-data//worker_2400.txt', 'r', encoding='utf-8') as f1:
            worker_dict = {}
            counttask = 0
            for line in f1.readlines():
                counttask += 1
                b = list(map(float, line.strip().split(' ')))
                worker_dict[counttask] = {
                    'Lw': b,
                    'Kw': sorted(random.sample(range(1, 11), 1))
                }
        return worker_dict

    def social_date_fun(self, worker_dict, history_task_range, task_history_number_range):
        task_history_quantity = [random.randint(*task_history_number_range) for _ in worker_dict]
        served_history_dict = {}
        for i, quantity in enumerate(task_history_quantity):
            task_history_served = sorted(random.sample(range(1, history_task_range + 1), quantity))
            served_history_dict[i + 1] = {"task_history_served": [quantity, task_history_served]}
        return served_history_dict

def real_cooperation_fun():
    with open('Real-data//cooperation_group.txt', 'r', encoding='utf-8') as f1:
        cooperation_dict = {}
        for line in f1.readlines():
            b = list(map(int, line.strip().split(' ')))
            if b[0] not in cooperation_dict:
                cooperation_dict[b[0]] = []
            cooperation_dict[b[0]].append(b[1])
    with open('cooperation_group.json', 'w', encoding='utf-8') as fp_cooperation:
        json.dump(cooperation_dict, fp_cooperation)
    return cooperation_dict

def real_worker_2700_fun(cooperation_dict):
    with open('Real-data//worker_2700.txt', 'r', encoding='utf-8') as f1:
        worker_dict = {}
        new_cooperation_dict = {}
        counttask = 0
        for line in f1.readlines():
            counttask += 1
            b = list(map(float, line.strip().split(' ')))
            worker_dict[counttask] = {
                'Lw': [b[1], b[2]],
                'Kw': sorted(random.sample(range(1, 11), 1))
            }
            if b[0] in cooperation_dict:
                new_cooperation_dict[counttask] = cooperation_dict[b[0]]

        cooperation_score_dict = {}
        for i in new_cooperation_dict:
            cooperation_score_dict[i] = [
                0.5 * 0.5 + 0.5 * len(set(new_cooperation_dict[i]).intersection(new_cooperation_dict.get(j, []))) / \
                len(set(new_cooperation_dict[i]).union(new_cooperation_dict.get(j, []))) if i != j else 0
                for j in new_cooperation_dict
            ]
        return worker_dict, cooperation_score_dict

def real_worker_fun():
    with open('Real-data//topic_id.txt', 'r', encoding='utf-8') as f1:
        topic_id = [list(map(int, line.strip().split(' '))) for line in f1.readlines()]
        final_topic_id = sorted(set(topic_id[0]).intersection(topic_id[1]))

    with open('Real-data//worker-topic.txt', 'r', encoding='utf-8') as f1:
        worker_skill_dict = {}
        for line in f1.readlines():
            b = list(map(int, line.strip().split(' ')))
            if b[1] in final_topic_id:
                worker_skill_dict.setdefault(b[0], []).append(b[1])

    with open('Real-data//worker_1000.txt', 'r', encoding='utf-8') as f1:
        worker_dict = {}
        for line in f1.readlines():
            b = list(map(float, line.strip().split(' ')))
            if int(b[0]) in worker_skill_dict:
                worker_dict[int(b[0])] = {
                    'Lw': [b[1], b[2]],
                    'Kw': worker_skill_dict[int(b[0])]
                }
        return worker_dict

def real_task_fun():
    with open('Real-data//topic_id.txt', 'r', encoding='utf-8') as f1:
        topic_id = [list(map(int, line.strip().split(' '))) for line in f1.readlines()]
        final_topic_id = sorted(set(topic_id[0]).intersection(topic_id[1]))

    with open('Real-data//task-topic.txt', 'r', encoding='utf-8') as f1:
        task_skill_dict = {}
        for line in f1.readlines():
            b = list(map(int, line.strip().split(' ')))
            task_skill_dict.setdefault(b[0], []).append(b[1])

    with open('Real-data//task_1000.txt', 'r', encoding='utf-8') as f1:
        task_dict = {}
        for line in f1.readlines():
            b = list(map(float, line.strip().split(' ')))
            if int(b[0]) in task_skill_dict:
                task_dict[int(b[0])] = {
                    'Lt': [b[1], b[2]],
                    'Kt': task_skill_dict[int(b[0])]
                }
        return task_dict

worker_dict = real_worker_fun()
task_dict = real_task_fun()
