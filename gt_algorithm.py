from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, satisfaction
from utils.distance_tool import Euclidean_fun
import copy


def check_nash_equilibrium(before_strategy, after_strategy):

    return before_strategy == after_strategy


def check_nash_equilibrium_ITS(task_dict, worker_dict, tasks, before_strategy, after_strategy, beta,v, alpha, worker_v,max_s, max_p):
    if -1 in before_strategy:
        return True
    before_utility = utility(task_dict, worker_dict, tasks, before_strategy,v, alpha, worker_v,max_s,max_p)
    after_utility = utility(task_dict, worker_dict, tasks, after_strategy,v, alpha,worker_v,max_s, max_p)
    aa=abs(after_utility - before_utility)
    return aa>beta*before_utility


def utility(task_dict, worker_dict, tasks, strategy, v, alpha,worker_v, max_s, max_p):
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
            u += task_utility_its(ai,task_dict, worker_dict, v,
                              task_pre_assign[ai]['list'], alpha, worker_v, max_s, max_p)
    return u

def price_score_its(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])

        return task_dict[t]['budget']*0.05 - dis * v
    return 0

def task_utility_its(ti, task_dict, worker_dict, v, w_list, alpha, worker_v, max_s,max_p):
    if len(w_list) == 0 or max_p == 0:
        return 0
    total_score = 0
    for w in w_list:
        total_score += worker_dict[w]['score']
    profit_w = price_score_its(task_dict, worker_dict, ti, v, w_list)

    return alpha * (total_score / len(w_list)) / max_s + (1 - alpha) * (profit_w / max_p)

def price_score(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
        # aa=len(worker_list)/ len(task_dict[t]['Kt'])
        return task_dict[t]['budget']/ len(task_dict[t]['Kt']) - dis * v * 10
    return 0

def task_utility(ti, task_dict, worker_dict, v, w_list, alpha, worker_v, max_s,max_p):
    if len(w_list) == 0 or max_p == 0:
        return 0
    min_score = 100
    total_score=0
    for w in w_list:
        total_score+= worker_dict[w]['score']
        if min_score>total_score:
            min_score=total_score
    profit_w = price_score(task_dict, worker_dict, ti, v, w_list)

    return alpha * (min_score) / max_s + (1 - alpha) * (profit_w / max_p)


def ability_check(worker, task):
    return True if worker['Kw'][0] in task['Kt'] else False


def gt(task_dict, worker_dict, workers, tasks, current_time, worker_v, v, alpha,max_s, max_p):
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
                                                              current_time, worker_v):
                space[wi].append(ti)
    before_strategy = {-1: -1}
    count=0
    while check_nash_equilibrium(before_strategy, strategy) is False:
        count+=1
        before_strategy=copy.deepcopy(strategy)
        for wi in workers:
            # select best task for wi(in the perspective of wi)
            best_task = 0
            best_task_conflict_w = 0
            best_list=[]
            utility_max = -1
            if strategy[wi]!=0:
                utility_max=task_utility(
                     strategy[wi], task_dict, worker_dict, v, task_pre_assign[strategy[wi]]['list'], alpha, worker_v, max_s, max_p)
            for ti in space[wi]:
                conflict_w = 0
                cur_list=[]
                # mod_strategy = copy.deepcopy(strategy)
                if len(task_pre_assign[ti]['list']) < len(task_dict[ti]['Kt']):
                    # add
                    if task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]] == 0:
                        cur_list.append(wi)
                        for w_pre in task_pre_assign[ti]['list']:
                            cur_list.append(w_pre)
                    else:
                        origin_wlist = task_pre_assign[ti]['list']
                        origin_u = task_utility(
                             ti, task_dict, worker_dict, v, origin_wlist, alpha,worker_v, max_s, max_p)
                        mod_wlist = []
                        conflict_w = task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]]
                        for w in origin_wlist:
                            if w != conflict_w:
                                mod_wlist.append(w)
                        mod_wlist.append(wi)
                        mod_u = task_utility(
                             ti, task_dict, worker_dict, v, mod_wlist, alpha,worker_v, max_s, max_p)
                        if mod_u > origin_u:
                            cur_list=mod_wlist
                        else:
                            continue
                else:
                    # replace
                    if task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]] != 0:
                        origin_wlist = task_pre_assign[ti]['list']
                        origin_u = task_utility(
                             ti, task_dict, worker_dict, v, origin_wlist, alpha,worker_v, max_s, max_p)
                        mod_wlist = []
                        conflict_w = task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]]
                        for w in origin_wlist:
                            if w != conflict_w:
                                mod_wlist.append(w)
                        mod_wlist.append(wi)
                        mod_u = task_utility(
                             ti, task_dict, worker_dict, v, mod_wlist, alpha,worker_v, max_s, max_p)
                        if mod_u > origin_u:
                            cur_list=mod_wlist
                        else:
                            continue
                cur_utility = task_utility(
                     ti, task_dict, worker_dict, v, cur_list, alpha, worker_v, max_s, max_p)
                if cur_utility > utility_max:
                    utility_max = cur_utility
                    best_task = ti
                    best_task_conflict_w = conflict_w
                    best_list=cur_list

            # change assign
            if best_task != 0:
                before_t=strategy[wi]
                if before_t!=0:
                    before_d_wi_list=set(task_pre_assign[before_t]['list'])
                    before_d_wi_list.discard(wi)
                    task_pre_assign[before_t]['list']=list(before_d_wi_list)

                strategy[wi] = best_task
                if(best_task_conflict_w in best_list):
                    count+=1
                    # print("at",wi,"find best_task_conflict_w in best_list!!")
                task_pre_assign[best_task]['list'] = best_list
                # print(best_task,task_pre_assign[best_task]['list'])
                task_pre_assign[best_task]['group'] = {}
                for ki in task_dict[best_task]['Kt']:
                    task_pre_assign[best_task]['group'][ki] = 0
                for w_b in best_list:
                    task_pre_assign[best_task]['group'][worker_dict[w_b]['Kw'][0]]=w_b
                if best_task_conflict_w != 0:
                    if strategy[best_task_conflict_w]!=0:
                        before_d_wi_list=set(task_pre_assign[strategy[best_task_conflict_w]]['list'])
                        before_d_wi_list.discard(best_task_conflict_w)
                        task_pre_assign[strategy[best_task_conflict_w]]['list']=list(before_d_wi_list)
                        if task_pre_assign[strategy[best_task_conflict_w]]['group'][worker_dict[best_task_conflict_w]['Kw'][0]]==best_task_conflict_w:
                            task_pre_assign[strategy[best_task_conflict_w]]['group'][worker_dict[best_task_conflict_w]['Kw'][0]]=0
                    strategy[best_task_conflict_w] = 0
        if count > 50:
            break

    return strategy, task_pre_assign

def gt_its(task_dict, worker_dict, workers, tasks, current_time, worker_v, v, alpha, max_p,max_s,beta):
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
                                                              current_time, worker_v):
                space[wi].append(ti)
    before_strategy = {-1: -1}
    count = 0
    while check_nash_equilibrium_ITS(task_dict,worker_dict,tasks,before_strategy, strategy,beta,v, alpha, worker_v,max_s, max_p):
        count += 1
        before_strategy=copy.deepcopy(strategy)
        for wi in workers:
            # select best task for wi(in the perspective of wi)
            best_task = 0
            best_task_conflict_w = 0
            best_list=[]
            utility_max = -1
            if strategy[wi]!=0:
                utility_max=task_utility_its(
                     strategy[wi], task_dict, worker_dict, v,task_pre_assign[strategy[wi]]['list'], alpha, worker_v,max_s, max_p)
            for ti in space[wi]:
                conflict_w = 0
                cur_list=[]
                # mod_strategy = copy.deepcopy(strategy)
                if len(task_pre_assign[ti]['list']) < len(task_dict[ti]['Kt']):
                    # add
                    if task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]] == 0:
                        cur_list.append(wi)
                        for w_pre in task_pre_assign[ti]['list']:
                            cur_list.append(w_pre)
                    else:
                        origin_wlist = task_pre_assign[ti]['list']
                        origin_u = task_utility_its(
                             ti, task_dict, worker_dict, v, origin_wlist, alpha, worker_v,max_s, max_p)
                        mod_wlist = []
                        conflict_w = task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]]
                        for w in origin_wlist:
                            if w != conflict_w:
                                mod_wlist.append(w)
                        mod_wlist.append(wi)
                        mod_u = task_utility_its(
                             ti, task_dict, worker_dict, v, mod_wlist, alpha, worker_v,max_s, max_p)
                        if mod_u > origin_u:
                            cur_list=mod_wlist
                        else:
                            continue
                else:
                    # replace
                    if task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]] != 0:
                        origin_wlist = task_pre_assign[ti]['list']
                        origin_u = task_utility_its(
                             ti, task_dict, worker_dict, v, origin_wlist, alpha, worker_v,max_s, max_p)
                        mod_wlist = []
                        conflict_w = task_pre_assign[ti]['group'][worker_dict[wi]['Kw'][0]]
                        for w in origin_wlist:
                            if w != conflict_w:
                                mod_wlist.append(w)
                        mod_wlist.append(wi)
                        mod_u = task_utility_its(
                             ti, task_dict, worker_dict, v, mod_wlist, alpha, worker_v,max_s, max_p)
                        if mod_u > origin_u:
                            cur_list=mod_wlist
                        else:
                            continue
                cur_utility = task_utility_its(
                     ti,task_dict, worker_dict, v,cur_list, alpha, worker_v,max_s, max_p)
                if cur_utility > utility_max:
                    utility_max = cur_utility
                    best_task = ti
                    best_task_conflict_w = conflict_w
                    best_list=cur_list

            # change assign
            if best_task != 0:
                before_t=strategy[wi]
                if before_t!=0:
                    before_d_wi_list=set(task_pre_assign[before_t]['list'])
                    before_d_wi_list.discard(wi)
                    task_pre_assign[before_t]['list']=list(before_d_wi_list)

                strategy[wi] = best_task
                if(best_task_conflict_w in best_list):
                    count += 1
                    print("at",wi,"find best_task_conflict_w in best_list!!")
                task_pre_assign[best_task]['list'] = best_list
                task_pre_assign[best_task]['group'] = {}
                for ki in task_dict[best_task]['Kt']:
                    task_pre_assign[best_task]['group'][ki] = 0
                for w_b in best_list:
                    task_pre_assign[best_task]['group'][worker_dict[w_b]['Kw'][0]]=w_b
                if best_task_conflict_w != 0:
                    if strategy[best_task_conflict_w]!=0:
                        before_d_wi_list=set(task_pre_assign[strategy[best_task_conflict_w]]['list'])
                        before_d_wi_list.discard(best_task_conflict_w)
                        task_pre_assign[strategy[best_task_conflict_w]]['list']=list(before_d_wi_list)
                        if task_pre_assign[strategy[best_task_conflict_w]]['group'][worker_dict[best_task_conflict_w]['Kw'][0]]==best_task_conflict_w:
                            task_pre_assign[strategy[best_task_conflict_w]]['group'][worker_dict[best_task_conflict_w]['Kw'][0]]=0
                    strategy[best_task_conflict_w] = 0
        if count>50:
            break
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


def gt_based_algorithm(task_dict, worker_dict, max_time, v, alpha, max_s, max_p, worker_v):
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
    skill_group = {}
    for i in task_dict.keys():
        skill_group[i] = {}
        d = [[] for i in range(len(task_dict[i]['Kt']))]
        for k in range(0, len(task_dict[i]['Kt'])):
            skill_group[i][task_dict[i]['Kt'][k]] = d[k]

    arrived_tasks = set()
    arrived_workers = set()
    sucess_to_assign_task = set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate
        # print("tick",current_time)
        # if current_time==16:
        #     print('at',16)
        # sucess_to_assign_task = set()
        sucess_to_assign_worker = set()
        after_discard_tasks = copy.deepcopy(arrived_tasks)
        for t_id in arrived_tasks:  # discard those meet their deadline
            if task_dict[t_id]['deadline'] >= current_time:
                after_discard_tasks.discard(t_id)
        arrived_tasks = after_discard_tasks
        after_discard_workers = copy.deepcopy(arrived_workers)
        for w_id in arrived_workers:  # discard those meet their deadline
            if worker_dict[w_id]['deadline'] >= current_time:
                after_discard_workers.discard(w_id)
        arrived_workers = after_discard_workers
        for t_id in task_dict.keys():  # add new
            if task_dict[t_id]['arrived_time'] <= current_time and t_id not in sucess_to_assign_task:
                arrived_tasks.add(t_id)
        for w_id in worker_dict.keys():  # add new
            if worker_dict[w_id]['arrived_time'] <= current_time:
                arrived_workers.add(w_id)
        workers = list(arrived_workers)
        tasks = get_free_task(task_dict, arrived_tasks, best_assign)

        equ_strategy, task_pre_assign = gt(
            task_dict, worker_dict, workers, tasks, current_time, worker_v,v,alpha,max_s,max_p)

        flag = 0
        for ai in task_pre_assign:
            if conflict_check(ai, task_assign_condition, task_dict) is False:
                if len(task_pre_assign[ai]['list']) == len(task_dict[ai]['Kt']):
                    flag += 1
                    task_assign_condition[ai]=current_time
                    best_assign[ai]['list'] = task_pre_assign[ai]['list']
                    best_assign[ai]['group'] = task_pre_assign[ai]['group']
                    best_assign[ai]['satisfaction'] = satisfaction(
                        task_dict, worker_dict, ai, v, best_assign[ai]['list'], alpha,worker_v, max_s, max_p)
                    best_assign[ai]['assigned'] = True
                    for w_assigned in best_assign[ai]['list']:
                        sucess_to_assign_worker.add(w_assigned)
                        if w_assigned not in worker_assign_time:
                            worker_assign_time[w_assigned]=[]
                        worker_assign_time[w_assigned].append({current_time:ai})
                        # if len(best_assign[ai]['list'])!=len(set(best_assign[ai]['list'])):
                        #     worker_assign_time[w_assigned].append('Err')
                    sucess_to_assign_task.add(ai)
                    print(ai,best_assign[ai]['list'],best_assign[ai]['satisfaction'])
        non_empty_list_count = sum(1 for value in task_pre_assign.values() if value['list'])
        print(non_empty_list_count)
        print('预分配:', flag)
        print(len(sucess_to_assign_task))
        print(current_time)
        arrived_tasks = arrived_tasks.difference(sucess_to_assign_task)
        arrived_workers = arrived_workers.difference(sucess_to_assign_worker)

    total_sat = 0
    for i in best_assign.keys():
        total_sat += best_assign[i]['satisfaction']
        if best_assign[i]['assigned'] == True:
            full_assign += 1
    print(len(sucess_to_assign_task))
    print(current_time)
    return task_assign_condition, best_assign, full_assign, total_sat,worker_assign_time


def gt_its_algorithm(task_dict, worker_dict, max_time, v, alpha, max_s, max_p, worker_v, beta):
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
    skill_group = {}
    for i in task_dict.keys():
        skill_group[i] = {}
        d = [[] for i in range(len(task_dict[i]['Kt']))]
        for k in range(0, len(task_dict[i]['Kt'])):
            skill_group[i][task_dict[i]['Kt'][k]] = d[k]

    arrived_tasks = set()
    arrived_workers = set()
    sucess_to_assign_task = set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate

        sucess_to_assign_worker = set()
        after_discard_tasks = copy.deepcopy(arrived_tasks)
        for t_id in arrived_tasks:  # discard those meet their deadline
            if task_dict[t_id]['deadline'] >= current_time:
                after_discard_tasks.discard(t_id)
        arrived_tasks = after_discard_tasks
        after_discard_workers = copy.deepcopy(arrived_workers)
        for w_id in arrived_workers:  # discard those meet their deadline
            if worker_dict[w_id]['deadline'] >= current_time:
                after_discard_workers.discard(w_id)
        arrived_workers = after_discard_workers
        for t_id in task_dict.keys():  # add new
            if task_dict[t_id]['arrived_time'] <= current_time and t_id not in sucess_to_assign_task:
                arrived_tasks.add(t_id)
        for w_id in worker_dict.keys():  # add new
            if worker_dict[w_id]['arrived_time'] <= current_time:
                arrived_workers.add(w_id)
        workers = list(arrived_workers)
        tasks = get_free_task(task_dict, arrived_tasks, best_assign)

        equ_strategy, task_pre_assign = gt_its(
            task_dict, worker_dict, workers, tasks, current_time, worker_v,v,alpha,max_p,max_s,beta)

        flag = 0
        for ai in task_pre_assign:
            if conflict_check(ai, task_assign_condition, task_dict) is False:
                if len(task_pre_assign[ai]['list']) == len(task_dict[ai]['Kt']):
                    flag += 1
                    task_assign_condition[ai]=current_time
                    best_assign[ai]['list'] = task_pre_assign[ai]['list']
                    best_assign[ai]['group'] = task_pre_assign[ai]['group']
                    best_assign[ai]['satisfaction'] = satisfaction(
                        task_dict, worker_dict, ai, v, best_assign[ai]['list'], alpha,worker_v, max_s, max_p)
                    best_assign[ai]['assigned'] = True
                    for w_assigned in best_assign[ai]['list']:
                        sucess_to_assign_worker.add(w_assigned)
                        if w_assigned not in worker_assign_time:
                            worker_assign_time[w_assigned]=[]
                        worker_assign_time[w_assigned].append({current_time:ai})

                    sucess_to_assign_task.add(ai)
                    print(ai,best_assign[ai]['list'],best_assign[ai]['satisfaction'])
        arrived_tasks = arrived_tasks.difference(sucess_to_assign_task)
        arrived_workers = arrived_workers.difference(sucess_to_assign_worker)

    total_sat = 0
    for i in best_assign.keys():
        total_sat += best_assign[i]['satisfaction']
        if best_assign[i]['assigned'] == True:
            full_assign += 1
    print('GT_ITS time:', total_sat)
    return task_assign_condition, best_assign, full_assign, total_sat,worker_assign_time