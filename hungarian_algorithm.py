import copy
from random import choice
from itertools import combinations, product
from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, Euclidean_fun, \
	satisfaction

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment

def temp_satisfaction(task_dict, worker_dict, t, move_cost, cooperation_workers, alpha, worker_v, max_s, max_p):

    if len(cooperation_workers) == 0 or max_p == 0:
        return 0
    total_score = 0
    for w in cooperation_workers:
        total_score += worker_dict[w]['score']
    profit_w = price_score(task_dict, worker_dict, t, move_cost, cooperation_workers)

    return alpha * (total_score / len(cooperation_workers)) / max_s + (1 - alpha) * (profit_w / max_p)


import numpy as np


def hungarian_algorithm_max_profit(profit_matrix):
	"""
    Hungarian algorithm to maximize total profit using SciPy's linear_sum_assignment.
    :param profit_matrix: NxM matrix of profits
    :return: optimal matching and total profit
    """
	# Convert the profit matrix to a numpy array
	profit_matrix = np.array(profit_matrix)

	# Step 1: Transform the profit matrix to a cost matrix
	max_value = profit_matrix.max()
	cost_matrix = max_value - profit_matrix

	# Step 2: Solve the assignment problem using linear_sum_assignment
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Step 3: Calculate total profit
	total_profit = profit_matrix[row_ind, col_ind].sum()

	# Step 4: Generate the optimal assignment
	matching = list(zip(row_ind, col_ind))

	return matching, total_profit



def get_order_in_related_tasks(dependency_graph, task):
	visited = set()
	dependent_tasks = []

	def dfs(current_task):
		visited.add(current_task)
		dependent_tasks.append(current_task)  # 将当前任务添加到结果列表
		for dependent_task in dependency_graph.get(current_task, []):
			if dependent_task not in visited:
				dfs(dependent_task)

	dfs(task)
	dependent_tasks.reverse()  # 反转结果列表
	return dependent_tasks


def get_order_related_tasks(dependency_graph, task):
	visited = set()
	dependent_tasks = []

	def dfs(current_task):
		visited.add(current_task)
		for dependent_task in dependency_graph.get(current_task, []):
			if dependent_task not in visited:
				dfs(dependent_task)
				dependent_tasks.append(dependent_task)

	dfs(task)
	dependent_tasks.append(task)
	return dependent_tasks

def init_bipartite(tasks, workers):
	G = nx.Graph()

	G.add_nodes_from(tasks, bipartite=0)
	G.add_nodes_from(workers, bipartite=1)

	return G


def price_score(task_dict, worker_dict, t, v, worker_list):
	if len(worker_list) > 0:
		dis = 0
		for i in worker_list:
			dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
		return task_dict[t]['budget'] - dis * v
	return 0



def price_score_be(task_dict, worker_dict, t, v, w):
	dis = 0
	dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])
	return task_dict[t]['budget'] - dis * v


def satisfaction_be(task_dict, worker_dict, ti, v, w, alpha, worker_v, max_s, max_p):
	total_score = worker_dict[w]['score']
	profit_w = price_score_be(task_dict, worker_dict, ti, v, w)

	return alpha * (total_score) / max_s + (1 - alpha) * (profit_w / max_p)


def init_best_assign(task_dict):
	best_assign = {}
	for i in task_dict.keys():
		best_assign[i] = {}
		best_assign[i]['list'] = []
		best_assign[i]['group'] = {}
		best_assign[i]['satisfaction'] = 0
		best_assign[i]['assigned'] = False
		for k in task_dict[i]['Kt']:
			best_assign[i]['group'][k] = 0
	return best_assign


def update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, sucess_to_assign_task,
					   delete_to_assign_task):

	for t_id in task_dict.keys():
		if task_dict[t_id]['deadline'] <= current_time or set(task_dict[t_id]['Ct']) & sucess_to_assign_task or tuple(
				task_dict[t_id]['Dt']) in delete_to_assign_task:
			delete_to_assign_task.add(t_id)

	for t_id in task_dict.keys():  # add new
		if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
				t_id not in sucess_to_assign_task and tuple(task_dict[t_id]['Ct']) not in sucess_to_assign_task \
				and tuple(task_dict[t_id]['Dt']) not in delete_to_assign_task:
			arrived_tasks.add(t_id)
	for w_id in worker_dict.keys():  # add new
		if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
			arrived_workers.add(w_id)
	return arrived_tasks, arrived_workers


def build_TC(task_dict, arrived_tasks):
	tl = sorted(list(arrived_tasks))
	TC = {}
	marked = {}
	for item in tl:
		marked[item] = False
	for _, t in enumerate(tl[::-1]):
		if marked[t] is True:
			continue
		task = task_dict[t]
		tc = []
		is_valid = True
		for depend_t in task['Dt']:
			if depend_t in arrived_tasks:
				tc.append(depend_t)
		if is_valid:
			tc.append(t)
			TC[t] = sorted(tc)
		marked[t] = True
	return TC

def get_order_in_related_tasks(dependency_graph, task):
    visited = set()
    dependent_tasks = []

    def dfs(current_task):
        visited.add(current_task)
        dependent_tasks.append(current_task)
        for dependent_task in dependency_graph.get(current_task, []):
            if dependent_task not in visited:
                dfs(dependent_task)

    dfs(task)
    dependent_tasks.reverse()
    return dependent_tasks

def get_available_task(task_dict, arrived_tasks, best_assign,current_time, TC, delete_to_assign_task):
    """
    """
    free_task = []
    for i in arrived_tasks:
        related_tasks = get_order_in_related_tasks(TC, i)
        is_free = True
        for depend_id in task_dict[i]['Dt']:
            if best_assign[depend_id]['assigned'] is False:
                is_free = False
                break
        for related_tasks_conflict in related_tasks:
            for id in task_dict[related_tasks_conflict]['Ct']:
                if best_assign[id]['assigned'] is True:
                    is_free = False
                    break
        if is_free:
            free_task.append(i)
    return free_task

def count(C):
	"""
    """
	return len(C)


def hungarian_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_p, worker_v, reachable_dis, max_s):
	# performance
	full_assign = 0

	# final task_workers assignment
	best_assign = init_best_assign(task_dict)

	skill_group = {}
	for i in task_dict.keys():
		skill_group[i] = {}
		d = [[] for i in range(len(task_dict[i]['Kt']))]
		for k in range(0, len(task_dict[i]['Kt'])):
			skill_group[i][task_dict[i]['Kt'][k]] = d[k]
	task_assign_condition = {}
	for i in task_dict.keys():
		task_assign_condition[i] = -1

	sucess_to_assign_task = set()
	delete_to_assign_task = set()
	for current_time in range(1, max_time+1):  # each time slice try to allocate
		arrived_tasks = set()
		arrived_workers = set()
		assigned_workers = set()
		sucess_to_assign_worker = set()

		arrived_tasks, arrived_workers = update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers,
															current_time, sucess_to_assign_task, delete_to_assign_task)


		TC = build_TC(task_dict, arrived_tasks)
		TC = {key: TC[key] for key in sorted(TC.keys())}

		tasks = get_available_task(task_dict, arrived_tasks, best_assign, current_time, TC, delete_to_assign_task)


		edges = []

		for tc in tasks:
			count_each_group_sa = {}
			if tc in sucess_to_assign_task:
				count_each_group_sa[tc[-1]] = 0
				continue
			if conflict_check(tc, task_assign_condition, task_dict):
				delete_to_assign_task.update(tc)
				TC_Group = {key: value for key, value in TC_Group.items() if
					  not any(num in value for num in tc)}
				break
			task = task_dict[tc]
			if len(task['Kt']) <= 1:
				for w_id in arrived_workers:
					if w_id not in sucess_to_assign_worker:
						worker = worker_dict[w_id]
						# distance check
						if satisfy_check(task, worker,
										 task['deadline'], worker['deadline'],
										 current_time, worker_v, move_cost, reachable_dis):
							if len(worker_dict[w_id]['Kw'])<=1:
									# skill_group[cur_task_id][task['Kt'][k]].append(w_id)
									cur_satisfaction = satisfaction_be(task_dict, worker_dict, tc,
																	   move_cost, w_id,
																	   alpha,
																	   worker_v, max_s, max_p)
									if cur_satisfaction < 0:
										cur_satisfaction = 0
									edges.append(
										(tc, w_id, cur_satisfaction))

		if edges:
			# Determine unique row and column indices
			cur_task_id = sorted(set(row for row, col, value in edges))
			cur_worker_id = sorted(set(col for row, col, value in edges))

			# Create a mapping for row and column indices to matrix indices
			row_mapping = {row: idx for idx, row in enumerate(cur_task_id)}
			col_mapping = {col: idx for idx, col in enumerate(cur_worker_id)}

			# Create an empty matrix of size (number of unique rows) x (number of unique columns) filled with zeros
			cost_matrix = np.zeros((len(cur_task_id), len(cur_worker_id)))

			# Populate the matrix using the data list
			for row, col, value in edges:
				cost_matrix[row_mapping[row]][col_mapping[col]] = value


			# cost_matrix[cost_matrix == float('-inf')] = np.min(cost_matrix[cost_matrix != float('-inf')]) - 1

			row_ind, col_ind = hungarian_algorithm_max_profit(cost_matrix) # Negate to maximize

			match_task_id = [pair[0] for pair in row_ind]
			match_worker_id = [pair[1] for pair in row_ind]

			count = 0
			for id in range(len(match_task_id)):
				cur_task = cur_task_id[match_task_id[id]]
				cur_worker = cur_worker_id[match_worker_id[id]]
				worker_list=[cur_worker]
				if conflict_check(cur_task, task_assign_condition, task_dict):
						delete_to_assign_task.discard(id)
						print("冲突")
				else:
					sucess_to_assign_task.add(cur_task)
					full_assign += 1
					best_assign[cur_task]['list'] = cur_worker
					for w in worker_list:

						sucess_to_assign_worker.add(w)
						assigned_workers.update(sucess_to_assign_worker)

					cur_task_satisfaction = satisfaction(task_dict, worker_dict, cur_task, move_cost, worker_list, alpha, worker_v,
														 max_s, max_p, current_time)
					best_assign[cur_task]['satisfaction'] = cur_task_satisfaction
					best_assign[cur_task]['assigned'] = True
					count+=1
					print(cur_task, worker_list,count)
					arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
					arrived_workers = arrived_workers.difference(
						set(sucess_to_assign_worker))

		for i in arrived_tasks:
			task = task_dict[i]

			# denpendency check
			if dependency_check(i, task_assign_condition, task_dict) is False:
				continue
			# check if conflict task is assigned
			if conflict_check(i, task_assign_condition, task_dict):
				continue
			candidate_worker = []
			for w_id in arrived_workers:
				worker = worker_dict[w_id]
				# distance check
				if satisfy_check(task, worker,
								 task['deadline'], worker['deadline'],
								 current_time, worker_v, move_cost, reachable_dis):
					candidate_worker.append(w_id)
			if len(candidate_worker) == 0:
				continue
			for k in range(0, len(task['Kt'])):
				for j in candidate_worker:
					for s in range(0, len(worker_dict[j]['Kw'])):
						if worker_dict[j]['Kw'][s] == task['Kt'][k]:
							skill_group[i][task['Kt'][k]].append(j)
			# print(i, skill_group[i])
			worker_list = []
			success_assign_flag = False
			for r in skill_group[i].keys():
				skill_list = list(
					set(skill_group[i][r]).difference(assigned_workers))
				if len(skill_list) != 0:
					"""greedy pick"""
					greedy_best_pick = -1
					greedy_best_sat = -1
					for current_worker in skill_list:
						worker_list.append(current_worker)
						current_sat = temp_satisfaction(task_dict, worker_dict, i, move_cost, worker_list, alpha,
														worker_v, max_s, max_p)
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
				sucess_to_assign_task.add(i)
				best_assign[i]['list'] = worker_list
				# NEED UPDATE
				for w in worker_list:
					best_assign[i]['group'][worker_dict[w]['Kw'][0]] = w
					sucess_to_assign_worker.add(w)
					assigned_workers.update(sucess_to_assign_worker)
				cur_task_satisfaction = satisfaction(task_dict, worker_dict, i, move_cost, worker_list, alpha, worker_v,
													 max_s, max_p, current_time)
				best_assign[i]['satisfaction'] = cur_task_satisfaction
				best_assign[i]['assigned'] = True
				print(i, worker_list)
				arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
				arrived_workers = arrived_workers.difference(
					set(sucess_to_assign_worker))




	total_sat = 0
	for i in best_assign.keys():
		total_sat += best_assign[i]['satisfaction']
	return task_assign_condition, best_assign, full_assign, total_sat



