import json, math, time, itertools, copy, random
from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, satisfaction
from utils.distance_tool import Euclidean_fun
from closest_algorithm import closest_algorithm
from gt_mutil_stage_algorithm import gt_base
from random import choice

def check_nash_equilibrium(before_strategy, after_strategy):

    return before_strategy == after_strategy


def check_nash_equilibrium_ITS(task_dict, worker_dict, tasks, before_strategy, after_strategy, beta,v, alpha, worker_v,max_s, max_p,TC,FT):
    if -1 in before_strategy:
        return True
    before_utility = utility(task_dict, worker_dict, tasks, before_strategy,v, alpha, worker_v,max_s,max_p,TC,FT)
    after_utility = utility(task_dict, worker_dict, tasks, after_strategy,v, alpha,worker_v,max_s, max_p,TC,FT)
    aa=abs(after_utility - before_utility)
    return aa>beta*before_utility



def utility(task_dict, worker_dict, tasks, strategy, v, alpha,worker_v, max_s, max_p,TC,FT):
    u = 0
    task_pre_assign = {}
    for ti in tasks:
        task_pre_assign[ti]={}
        task_pre_assign[ti]['list'] = []
        task_pre_assign[ti]['group'] = {}
        for ki in task_dict[ti]['Kt']:
            task_pre_assign[ti]['group'][ki] = 0

    for i in strategy:
        if(strategy[i]==0):
            continue
        task_pre_assign[strategy[i]]['list'].append(i)
        task_pre_assign[strategy[i]]['group'][worker_dict[i]['Kw'][0]] = i
    for ai in task_pre_assign:
        if len(task_pre_assign[ai]['list']) != len(task_dict[ai]['Kt']):
            continue
        flag = True
        groups = task_pre_assign[ai]['group']
        for g in groups:
            if groups[g] == 0:
                flag = False
                break
        if flag:
            u += task_utility(ai,task_dict, worker_dict, v,
                              task_pre_assign[ai]['list'], alpha, worker_v, max_s, max_p,TC,FT)

    return u

def price_score_imp1(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) #0.42
        cur_task_cost=task_dict[t]['budget'] #- dis * v

    return cur_task_cost*0.05- dis * v


def price_score_imp(task_dict, worker_dict, t, v, worker_list, related_tasks):
    avg_task_cost = 0
    # result = w_skill // 10

    for i in related_tasks:
        avg_task_cost+=task_dict[i]['budget']/(len(task_dict[i]['Kt']))
    if related_tasks:
        avg_task_cost = avg_task_cost/len(related_tasks)
    first_task_cost=task_dict[t]['budget']/(len(task_dict[t]['Kt']))
    task_cost=max(avg_task_cost, first_task_cost)

    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) #0.42
    # dis=dis/(len(worker_list))
    return task_cost- dis * v

def task_utility(ti, task_dict, worker_dict, v, w_list, alpha, worker_v, max_s,max_p,TC,FT):
    related_tasks= copy.deepcopy(FT[ti][:0])

    if len(w_list) == 0 or max_p == 0:
        return 0
    total_score=0
    for w in w_list:
        total_score+= worker_dict[w]['score']/(len(task_dict[ti]['Kt']))

    total_score = total_score/(len(w_list))
    profit_w = price_score_imp(task_dict, worker_dict, ti, v, w_list, related_tasks)
    cur_total=alpha * (total_score) / max_s + (1 - alpha) * (profit_w / max_p)
    return cur_total

def random_strategy(task_dict, worker_dict, workers, tasks,
                    current_time, worker_v, v, alpha,max_s, max_p,move_cost,reachable_dis,TC,
                    task_pre_assign,strategy):

    skill_group = {}
    for i in tasks:
        skill_group[i] = {}
        d = [[] for i in range(len(task_dict[i]['Kt']))]
        for k in range(0, len(task_dict[i]['Kt'])):
            skill_group[i][task_dict[i]['Kt'][k]] = d[k]

    task_assign_condition = {}
    for i in task_dict.keys():
        task_assign_condition[i] = -1

    assigned_workers=set()
    assigned_tasks = set()
    _assigned_workers=set()
    for task_id in task_pre_assign:
        # print(i)
        task = task_dict[task_id]
        related_tasks = get_order_related_tasks(TC, task_id)
        if len(related_tasks)>0:
            continue
        # denpendency check
        if dependency_check(task_id, task_assign_condition, task_dict) is False:
            continue
        # check if conflict task is assigned
        if conflict_check(task_id, task_assign_condition, task_dict):
            continue
        candidate_worker = []
        for w_id in workers:
            worker = worker_dict[w_id]
            # distance check
            if satisfy_check(task, worker,
                                task['deadline'], worker['deadline'],
                                current_time, worker_v, move_cost, reachable_dis):
                candidate_worker.append(w_id)
        if len(candidate_worker) == 0:
            break
        for k in range(0, len(task['Kt'])):
            for j in candidate_worker:
                for s in range(0, len(worker_dict[j]['Kw'])):
                    if worker_dict[j]['Kw'][s] == task['Kt'][k]:
                        skill_group[task_id][task['Kt'][k]].append(j)
        worker_list = []
        success_assign_flag = False
        for r in skill_group[task_id].keys():
            assigned_workers.update(_assigned_workers)
            skill_list = list(
                set(skill_group[task_id][r]).difference(assigned_workers))
            if len(skill_list) != 0:
                """random pick a worker"""
                worker_w = choice(skill_list)
                worker_list.append(worker_w)
                task_pre_assign[task_id]['list'].append(worker_w)
                strategy[worker_w]=task_id
                task_pre_assign[task_id]['group'][r]=worker_w
                _assigned_workers.add(worker_w)
                task_assign_condition[task_id]=0

    return task_pre_assign, strategy


def build_TC(task_dict,arrived_tasks):
    tl=sorted(list(arrived_tasks))
    TC={}
    marked={}
    for item in tl:
        marked[item]=False
    for _, t in enumerate(tl[::-1]):
        if marked[t] is True:
            continue
        task=task_dict[t]
        tc=[]
        is_valid=True
        for depend_t in task['Dt']:
            if depend_t in arrived_tasks:
                tc.append(depend_t)
        if is_valid:
            tc.append(t)
            TC[t]=sorted(tc)
        marked[t]=True
    return TC


def get_dependency_set(task_dict, task_id):
    dependencies = set()

    visited_tasks = set()

    def find_dependencies(task, level):
        if task in visited_tasks:
            return
        visited_tasks.add(task)

        for t, info in task_dict.items():
            if task in info['Dt']:
                dependencies.add(t)
                find_dependencies(t, level + 1)

    find_dependencies(task_id, 1)
    sorted_dependencies = sorted(list(dependencies), key=lambda x: task_dict[x]['budget'], reverse=True)

    return sorted_dependencies

def build_FC(task_dict, task_ids):
    all_dependencies = {}
    for task_id in task_ids:
        all_dependencies[task_id] = get_dependency_set(task_dict, task_id)
    return all_dependencies

def build_FT(task_dict, arrived_tasks, best_assign):
    tl = sorted(list(arrived_tasks))
    FT = {}
    for t in tl:
        FT[t] = []
    for t in tl:
        task = task_dict[t]
        for depend_t in task['Dt']:
            if depend_t in arrived_tasks:
                FT[depend_t].append(t)
    return FT_helper(task_dict, arrived_tasks, FT)


def FT_helper(task_dict, arrived_tasks, FT_direct):
    FT = {}
    FT_rank = {}
    for t in arrived_tasks:
        visited = set()
        ft = []
        ft_rank = []
        def dfs(t, rank):
            visited.add(t)
            ft.append(t)
            ft_rank.append(rank)
            for future_task in FT_direct.get(t, []):
                if future_task not in visited:
                    dfs(future_task, rank + 1)

        dfs(t, 1)
        FT[t] = ft
        FT_rank[t] = ft_rank
    return FT, FT_rank

def get_related_tasks(dependency_graph, task):
    visited = set()
    dependent_tasks = set()

    def dfs(current_task):
        visited.add(current_task)
        for dependent_task in dependency_graph.get(current_task, []):
            if dependent_task not in visited:
                dfs(dependent_task)
                dependent_tasks.add(dependent_task)
    dfs(task)
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
    # dependent_tasks.append(task)
    return dependent_tasks

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

def ability_check(worker, task):
    return True if set(worker['Kw']).intersection(set(task['Kt'])) else False

def gt_imp(task_dict, worker_dict, workers, tasks, current_time, worker_v,reachable_dis, move_cost, alpha,max_s, max_p,TC,FT,beta):
    strategy = {}
    task_pre_assign = {}
    for ti in tasks:
        task_pre_assign[ti]={}
        task_pre_assign[ti]['list'] = []
        task_pre_assign[ti]['group'] = {}
        for ki in task_dict[ti]['Kt']:
            task_pre_assign[ti]['group'][ki] = 0
    for wi in workers:
        strategy[wi] = 0
    space = {}
    for wi in workers:
        space[wi] = []
        worker = worker_dict[wi]
        for ti in tasks:
            task = task_dict[ti]
            if ability_check(worker, task) and satisfy_check(task, worker,
                                                              task['deadline'], worker['deadline'],
                                                              current_time, worker_v,move_cost,reachable_dis):
                space[wi].append(ti)
    before_strategy = {-1: -1}
    count=0
    _assigned_task= set()

    task_pre_assign, strategy = random_strategy(task_dict, worker_dict, workers, tasks,
                                      current_time, worker_v, move_cost, alpha,max_s, max_p,move_cost,reachable_dis,TC,
                                      task_pre_assign,strategy)


    # task_pre_assign,strategy = greedy_strategy(task_dict, worker_dict, workers, tasks,
    #                                   current_time, worker_v, move_cost, alpha,max_s, max_p,move_cost,reachable_dis,TC,
    #                                   task_pre_assign,strategy)

    while check_nash_equilibrium_ITS(task_dict, worker_dict, tasks, before_strategy, strategy, beta, move_cost, alpha,
                                     worker_v, max_s, max_p, TC, FT):
        count+=1
        # print(count)
        before_strategy=copy.deepcopy(strategy)
        for wi in workers:
            # select best task for wi(in the perspective of wi)
            best_task = 0
            best_skill = 0
            best_task_conflict_w = 0
            best_list=[]
            utility_max = -1
            if strategy[wi]!=0:
                group = copy.deepcopy(task_pre_assign[strategy[wi]])
                target_value = wi
                w_s = [key for key, value in group['group'].items() if value == target_value]
                utility_max = task_utility(
                    strategy[wi], task_dict, worker_dict, move_cost, [wi],  alpha, worker_v, max_s, max_p, TC,FT)
            for ti in space[wi]:
                related_tasks = get_order_related_tasks(TC, ti)
                all_elements_in_set = all(element in _assigned_task for element in tuple(related_tasks))
                # all_elements_in_set=True
                if all_elements_in_set:
                    # mod_strategy = copy.deepcopy(strategy)
                    for w_skill in worker_dict[wi]['Kw']:
                        conflict_w = 0
                        cur_list = []
                        cur_skill = []
                        group = copy.deepcopy(task_pre_assign[ti])
                        if any(wi == int(value) for value in task_pre_assign[ti]['group'].values()) is False:
                            if w_skill in task_pre_assign[ti]['group']:

                                if len(task_pre_assign[ti]['list']) < len(task_pre_assign[ti]['group']):
                                    # ad
                                    if task_pre_assign[ti]['group'][w_skill] == 0:

                                        cur_list.append(wi)
                                        cur_skill=w_skill
                                        cur_worker = wi
                                        group['group'][w_skill]=wi
                                        for w_pre in task_pre_assign[ti]['list']:
                                            cur_list.append(w_pre)
                                    else:
                                        conflict_w = task_pre_assign[ti]['group'][w_skill]
                                        origin_wlist=task_pre_assign[ti]['list']
                                        origin_u = task_utility(
                                             ti, task_dict, worker_dict, move_cost, [conflict_w], alpha,worker_v, max_s, max_p,TC,FT)
                                        mod_wlist = []
                                        # conflict_w = task_pre_assign[ti]['group'][w_skill]
                                        for w in origin_wlist:
                                            if w != conflict_w:
                                                mod_wlist.append(w)
                                        group['group'][w_skill] = wi
                                        mod_wlist.append(wi)
                                        mod_u = task_utility(
                                             ti, task_dict,worker_dict, move_cost, [wi], alpha,worker_v, max_s, max_p,TC,FT)
                                        conflict_u = 0
                                        if set(task_dict[ti]['Ct']) in _assigned_task:
                                            conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                            conflict_u = task_utility(
                                                ti, task_dict, _assigned_task, task_pre_assign, worker_dict, group,
                                                move_cost, conflict_wlist, w_skill, alpha, worker_v, max_s, max_p, TC,
                                                FT)
                                        if mod_u <= conflict_u:
                                            break

                                        if mod_u > origin_u:
                                            cur_skill=w_skill
                                            cur_worker=wi
                                            cur_list=mod_wlist
                                        else:
                                            continue
                                else:
                                    conflict_u = 0
                                    if set(task_dict[ti]['Ct']) in _assigned_task:
                                        conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                        conflict_u = task_utility(
                                            ti, task_dict,  worker_dict,
                                            move_cost, conflict_wlist,  alpha, worker_v, max_s, max_p, TC, FT)

                                    # replace
                                    if task_pre_assign[ti]['group'][w_skill] != 0:
                                        origin_wlist = task_pre_assign[ti]['list']
                                        conflict_w = task_pre_assign[ti]['group'][w_skill]
                                        origin_u = task_utility(
                                             ti, task_dict,worker_dict, move_cost, [conflict_w], alpha,worker_v, max_s, max_p,TC,FT)
                                        mod_wlist = []
                                        # conflict_w = task_pre_assign[ti]['group'][w_skill]
                                        for w in origin_wlist:
                                            if w != conflict_w:
                                                mod_wlist.append(w)
                                        group['group'][w_skill] = wi
                                        mod_wlist.append(wi)
                                        mod_u = task_utility(
                                             ti, task_dict, worker_dict, move_cost, [wi],  alpha,worker_v, max_s, max_p,TC,FT)

                                        conflict_u = 0
                                        if set(task_dict[ti]['Ct']) in _assigned_task:
                                            conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                            conflict_u = task_utility(
                                                ti, task_dict,  worker_dict,
                                                move_cost, conflict_wlist,  alpha, worker_v, max_s, max_p, TC,
                                                FT)
                                        if mod_u <= conflict_u:
                                            break

                                        if mod_u > origin_u:
                                            cur_skill=w_skill
                                            cur_worker=wi
                                            cur_list=mod_wlist
                                        else:
                                            continue
                                # print(group)
                                cur_utility = task_utility(
                                     ti, task_dict, worker_dict ,  move_cost, [cur_worker],  alpha, worker_v, max_s, max_p,TC,FT)

                                conflict_u=0
                                if set(task_dict[ti]['Ct']) in _assigned_task:
                                    conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                    conflict_u = task_utility(
                                        ti, task_dict,  worker_dict,
                                        move_cost, conflict_wlist,  alpha, worker_v, max_s, max_p, TC, FT)

                                if cur_utility<=conflict_u:
                                    cur_utility=0

                                if cur_utility > utility_max:
                                    utility_max = cur_utility
                                    best_task = ti
                                    best_worker = cur_worker
                                    best_skill=cur_skill
                                    best_task_conflict_w = conflict_w
                                    conflict_skill=cur_skill
                                    # if best_task_conflict_w!=0 and best_task!=strategy[best_task_conflict_w]:
                                    #     print("at",wi,"find best_task!=strategy[best_task_conflict_w]",(best_task,strategy[best_task_conflict_w]))
                                    best_list=cur_list

            if best_task != 0:
                before_t = strategy[wi]
                if before_t != 0:
                    before_d_wi_list = set(task_pre_assign[before_t]['list'])
                    before_d_wi_list.discard(wi)
                    task_pre_assign[before_t]['list'] = list(before_d_wi_list)
                    desired_value = wi
                    c_skill = [key for key, value in task_pre_assign[before_t]['group'].items() if
                               value == desired_value]
                    task_pre_assign[before_t]['group'][c_skill[0]] = 0

                strategy[wi] = best_task
                if (best_task_conflict_w in best_list):
                    count += 1
                    # print("at",wi,"find best_task_conflict_w in best_list!!")
                _assigned_task.add(best_task)
                task_pre_assign[best_task]['list'] = best_list
                # print(best_task, task_pre_assign[best_task]['list'])
                task_pre_assign[best_task]['group'][conflict_skill] = best_worker
                if best_task_conflict_w != 0:
                    if strategy[best_task_conflict_w] != 0:
                        before_d_wi_list = set(task_pre_assign[strategy[best_task_conflict_w]]['list'])
                        before_d_wi_list.discard(best_task_conflict_w)
                        task_pre_assign[strategy[best_task_conflict_w]]['list'] = list(before_d_wi_list)

                        if task_pre_assign[strategy[best_task_conflict_w]]['group'][
                            conflict_skill] == best_task_conflict_w:
                            task_pre_assign[strategy[best_task_conflict_w]]['group'][conflict_skill] = 0
                    strategy[best_task_conflict_w] = 0

    return strategy, task_pre_assign


def get_free_task(task_dict, arrived_tasks, best_assign):
    """
    """
    free_task = []
    for i in arrived_tasks:
        is_free = True
        for depend_id in task_dict[i]['Dt']:
            if best_assign[depend_id]['assigned'] is False:
                is_free = False
                break
        for conflict_id in task_dict[i]['Ct']:
            if best_assign[conflict_id]['assigned'] is True:
                is_free = False
                break
        if is_free:
            free_task.append(i)
    return free_task

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
        # for related_tasks_id in related_tasks:
        #     for id in task_dict[related_tasks_conflict]['Ct']:
        # if related_tasks.intersection(delete_to_assign_task) is True:
        #     is_free = False
        if len(related_tasks)>10:
            is_free = False
        if is_free:
            free_task.append(i)
    return free_task

def gt_jdcta_its_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_s, max_p, worker_v,reachable_dis,beta):
    # performance
    full_assign = 0
    worker_assign_time={}
    task_assign_condition = {}
    for i in task_dict.keys():
        task_assign_condition[i] = -1
    # final task_workers assignment
    best_assign = {}
    for i in task_dict.keys():
        best_assign[i] = {}
        best_assign[i]['list'] = []
        best_assign[i]['group'] = {}
        best_assign[i]['satisfaction'] = 0
        best_assign[i]['assigned'] = False
        for k in task_dict[i]['Kt']:
            best_assign[i]['group'][k] = 0

    sucess_to_assign_task = set()
    delete_to_assign_task = set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate
        arrived_tasks = set()
        arrived_workers = set()
        if current_time<max_time+1:
            # arrived_tasks = set()
            # arrived_workers = set()
            sucess_to_assign_worker = set()
            # after_discard_tasks = copy.deepcopy(arrived_tasks)
            for t_id in task_dict.keys():  # discard those meet their deadline set(task_dict[t_id]['Ct']).issubset(sucess_to_assign_task)
                if task_dict[t_id]['deadline'] < current_time or set(
                        task_dict[t_id]['Ct']) & sucess_to_assign_task or tuple(
                        task_dict[t_id]['Dt']) in delete_to_assign_task:
                    delete_to_assign_task.add(t_id)
                    # arrived_tasks = arrived_tasks.remove(t_id)
                    # arrived_workers = arrived_workers.difference(t_id)
            for t_id in task_dict.keys():  # add new
                if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
                        t_id not in sucess_to_assign_task and tuple(task_dict[t_id]['Ct']) not in sucess_to_assign_task \
                        and tuple(
                    task_dict[t_id]['Dt']) not in delete_to_assign_task and t_id not in delete_to_assign_task:
                    arrived_tasks.add(t_id)
            for w_id in worker_dict.keys():  # add new
                if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
                    arrived_workers.add(w_id)

            flag = 1
            while flag != 0:
                workers = list(arrived_workers)
                TC = build_TC(task_dict, arrived_tasks)
                tasks = get_free_task(task_dict, arrived_tasks, best_assign)
                FC = build_FC(task_dict, tasks)

                equ_strategy, task_pre_assign = gt_imp(
                    task_dict, worker_dict, workers, tasks, current_time, worker_v,reachable_dis,move_cost,alpha,max_s,max_p,TC,FC,beta)

                flag = 0
                for ai in task_pre_assign:
                    # print(ai, task_pre_assign[ai]['list'])
                    if conflict_check(ai, task_assign_condition, task_dict) is False:
                        satisfied = True
                        for task_id in task_dict[ai]['Dt']:
                            if task_assign_condition[task_id] == -1:
                                satisfied = False
                        if satisfied!=False:
                            if len(task_pre_assign[ai]['list']) == len(task_dict[ai]['Kt']) and set(task_pre_assign[ai]['list']).issubset(sucess_to_assign_worker) is False:
                                flag += 1
                                best_assign[ai]['list'] = task_pre_assign[ai]['list']
                                best_assign[ai]['group'] = task_pre_assign[ai]['group']
                                best_assign[ai]['satisfaction'] = satisfaction(
                                    task_dict, worker_dict, ai, move_cost, best_assign[ai]['list'], alpha,worker_v, max_s, max_p,current_time)
                                best_assign[ai]['assigned'] = True
                                for w_assigned in best_assign[ai]['list']:
                                    sucess_to_assign_worker.add(w_assigned)
                                    if w_assigned not in worker_assign_time:
                                        worker_assign_time[w_assigned]=[]
                                    worker_assign_time[w_assigned].append({current_time:ai})

                                task_assign_condition[ai] = current_time
                                sucess_to_assign_task.add(ai)
                                # print(ai,best_assign[ai]['list'],best_assign[ai]['satisfaction'])

                non_empty_list_count = sum(1 for value in task_pre_assign.values() if value['list'])
                total_sat = 0
                for i in best_assign.keys():
                    total_sat += best_assign[i]['satisfaction']

                arrived_tasks = arrived_tasks.difference(sucess_to_assign_task)
                arrived_workers = arrived_workers.difference(sucess_to_assign_worker)
            else:
                workers = list(arrived_workers)
                tasks = get_free_task(task_dict, arrived_tasks, best_assign)

                equ_strategy, task_pre_assign = gt_base(
                    task_dict, worker_dict, workers, tasks, current_time, worker_v, reachable_dis, move_cost, alpha, max_s, max_p)

                for ai in task_pre_assign:
                    if conflict_check(ai, task_assign_condition, task_dict) is False:
                        satisfied = True
                        for task_id in task_dict[ai]['Dt']:
                            if task_assign_condition[task_id] == -1:
                                satisfied = False
                        if satisfied != False:
                            if len(task_pre_assign[ai]['list']) == len(task_dict[ai]['Kt']) and set(
                                    task_pre_assign[ai]['list']).issubset(sucess_to_assign_worker) is False:
                                best_assign[ai]['list'] = task_pre_assign[ai]['list']
                                best_assign[ai]['group'] = task_pre_assign[ai]['group']
                                best_assign[ai]['satisfaction'] = satisfaction(
                                    task_dict, worker_dict, ai, move_cost, best_assign[ai]['list'], alpha, worker_v, max_s, max_p,
                                    current_time)
                                best_assign[ai]['assigned'] = True
                                flag += 1
                                for w_assigned in best_assign[ai]['list']:
                                    sucess_to_assign_worker.add(w_assigned)
                                    if w_assigned not in worker_assign_time:
                                        worker_assign_time[w_assigned] = []
                                    worker_assign_time[w_assigned].append({current_time: ai})

                                task_assign_condition[ai] = current_time
                                sucess_to_assign_task.add(ai)
                                # print(ai, best_assign[ai]['list'], best_assign[ai]['satisfaction'])

    total_sat = 0
    for i in best_assign.keys():
        total_sat += best_assign[i]['satisfaction']
        if best_assign[i]['assigned'] == True:
            full_assign += 1
    return task_assign_condition, best_assign, full_assign, total_sat,worker_assign_time
