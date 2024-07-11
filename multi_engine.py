import json 
import copy 
import numpy as np
import itertools
from multiprocessing import Pool
import time
import os
from typing import List
from bo import BayesianOptimization
from data import Query, QuerySetup, TierSpec, SetupMetric
from cluster import ClusterState
from task import speech_recognition_ops, speech_recongnition_knobs, Knob, Operator, speech_recognition_dnn_usg
from utils import get_feat_bounds, probability_greater_than_t, sample_to_config, config_to_sample

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Engine:
    def __init__(self, queries: List[Query], cluster_spec: List[TierSpec], num_profile_sample=-1):
        # BO arguments
        self.early_prune_count = 10 
        self.bo_stop_count = 3
        self.num_init_sample_configs = 5
        self.utility_thold = 0  

        # metric to evalute BO efficiency
        self.num_profile_sample = num_profile_sample
        self.cache = {}
        self.profile_lat = 0        

        # Engine Spec (Machine & Quries)
        self.queries = queries
        self.cluster_spec = cluster_spec
        
        # Speed up by loading all result file.
        self.profile_result_cache = {}
        for query in self.queries:
            if query.profile_result_file not in self.profile_result_cache:
                with open(query.profile_result_file, 'r') as fp:
                    profile_result = json.load(fp)        
                self.profile_result_cache[query.profile_result_file] = profile_result   
             


    def simulate_pipeline(self, config: dict, query: Query) -> dict:

        config_str = json.dumps(config)
        if config_str in self.cache:
            return self.cache[config_str]

        profile_result = self.profile_result_cache[query.profile_result_file]
        for pf in profile_result:
            found = True
            for knob_name, knob_val in config.items():
                if pf[knob_name] != knob_val:
                    found = False
                    break
            if found:
                self.cache[config_str] = pf["result"]
                self.profile_lat += pf["result"]["total_profile_time"] * (self.num_profile_sample / len(pf["result"]["cummulative_accuracy"]))
                return pf["result"]

        return {}

    def postprocess_result(self, profile_result: dict, placement: List[int], config: dict, query: Query) -> SetupMetric:
        prev_op_machine = 0
        tot_compute_latency = 0
        tot_transfer_latency = 0
        tot_gpu_time = 0
        compute_latency_by_op = []
        gpu_mem_usg_by_op = []
        for op, op_machine in zip(query.op_list, placement):
            op_name = op.name

            compute_latency = profile_result[f"{op_name}_compute_latency"] / mach_info['compute'][op_machine]
            compute_latency_by_op.append(compute_latency)

            # TODO: Ensure this assumption is correct:
            if op.tpy == 'dnn':
                gpu_mem_usg_by_op.append(query.dnn_gpu_usg[config['model']])
                tot_gpu_time += profile_result[f"{op_name}_compute_latency"] # tot_gpu_time processing by "profiled" machine 
            else:
                gpu_mem_usg_by_op.append(0)

            transfer_latency = 0
            for machine_id in range(prev_op_machine, op_machine):
                bandwidth = mach_info["bandwidth"][machine_id] #gbs
                bandwidth *= 10e9 #bytes/s
                transfer_latency += profile_result[f"{op_name}_input_size"] / bandwidth
            
            tot_compute_latency += compute_latency
            tot_transfer_latency += transfer_latency

        pipe_accuracy = profile_result["cummulative_accuracy"][self.num_profile_sample - 1]
        pipe_latency = tot_compute_latency + tot_transfer_latency

        utility = (0.5 * -1 * (pipe_accuracy - query.acc_thold) + 0.5 * (query.lat_thold - pipe_latency)) / tot_gpu_time
        acc_index = -1 * (pipe_accuracy - query.acc_thold) / tot_gpu_time

        setup_metric = SetupMetric(
            accuracy=pipe_accuracy,
            latency=pipe_latency,
            utility=utility,
            tot_transfer_latency=tot_transfer_latency,
            tot_gpu_time=tot_gpu_time,
            accuracy_index=acc_index,
            compute_latency_by_op=compute_latency_by_op,
            gpu_mem_usg_by_op=gpu_mem_usg_by_op
        )
        
       
        return setup_metric
    
    def generate_placement_choices(self, query: Query, cluster_spec: List[TierSpec]) -> List[List[int]]:
        num_ops = len(query.op_list)
        num_tiers = len(cluster_spec)

        poss_plcs = []

        def dfs(cur_plc):
            if len(cur_plc) >= num_ops:
                poss_plcs.append(cur_plc)
                return

            prev_op_plc = 0
            if len(cur_plc) > 0:
                prev_op_plc = cur_plc[-1]
            for p in range(prev_op_plc, num_tiers):
                dfs(cur_plc + [p])
        dfs([])

        # # filter choices by cluster state.
        # for plc in poss_plcs:
        #     # check feasibiliy
        #     cluster_state.can_place_pipline(plc, query.op_list) 


        return poss_plcs 
    
    def generate_all_configs(self, query):
        num_knobs = len(query.knob_list)

        all_choices = []

        def dfs(cur_config):
            cur_num_knob = len(cur_config)
            if cur_num_knob >= num_knobs:
                all_choices.append(cur_config)
                return

            knob_name = query.knob_list[cur_num_knob].name
            for knob_choice in query.knob_list[cur_num_knob].choice:
                cur_config[knob_name] = knob_choice
                dfs(copy.deepcopy(cur_config))
                cur_config.pop(knob_name)

        dfs({})
        return all_choices

    def search_by_utlity(self, query: Query, cluster_state: ClusterState, optimize: bool=False):
        """
        Search a query_setup for a query using BO-utiltiy  
            vulcan algorithm
        """

        self.cache = {}
        self.profile_lat = 0
        feat_bounds = get_feat_bounds(query) 
        placements = self.generate_placement_choices(query, cluster_state.spec)

        query_setups: List[QuerySetup] = []
        # total_profile_time = 0
        for placement in placements:
            bo = BayesianOptimization(feat_bounds=feat_bounds) 
            init_utils = [] # utility

            # opt: using history by other placements
            init_samples: List[List[int]] = []
            init_utils: List[float] = []

            # BO Intialization 
            if optimize and len(self.cache) >= 3 :
                for config_str, result in self.cache.items():
                    config = json.loads(config_str)
                    setup_metric = self.postprocess_result(result, placement, config, query)
                    sample = config_to_sample(config, query.knob_list) 
                    #
                    init_samples.append(sample)
                    init_utils.append(setup_metric.utility)
                    # utilities.append((utility, appox_latency, result["accuracy"], config, placement, 'prev'))

                bo.fit(X=np.array(init_samples), y=np.array(init_utils))
            else: 
                # 1. Random sample configs
                init_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
                init_utils = []
                # 2. Run those config to get ACC & Latency
                for sample in init_samples:
                    
                    config = sample_to_config(sample, query.knob_list)
                    profile_result = self.simulate_pipeline(config=config, query=query)  
                    setup_metric = self.postprocess_result(profile_result, placement, config, query)

                    init_utils.append(setup_metric.utility)
                    query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))

                bo.fit(X=np.array(init_samples), y=np.array(init_utils))

            # BO Searching
            prune_count, stop_count = 0, 0  
            max_util = max(init_utils)
            while True:
                sample = bo.get_next_sample()
                config = sample_to_config(sample, query.knob_list) 
                profile_result = self.simulate_pipeline(config=config, query=query)
                setup_metric = self.postprocess_result(profile_result, placement, config, query) 


                # Fit BO again
                init_samples.append(sample)
                init_utils.append(setup_metric.utility)
                bo.fit(X=np.array(init_samples), y=np.array(init_utils))

                query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))

                prune_count += 1 if setup_metric.utility < self.utility_thold else 0
                stop_count += 1 if setup_metric.utility < max_util else 0

                max_util = max(setup_metric.utility, max_util)

                if prune_count >= self.early_prune_count or stop_count > self.bo_stop_count:
                    break
        
        return query_setups  

    def search_by_accuracy(self, query: Query, cluster_state: ClusterState):
        feat_bounds = get_feat_bounds(query)
        placements = self.generate_placement_choices(query, self.cluster_spec)
        placement = placements[0]
        bo = BayesianOptimization(feat_bounds=feat_bounds) 

        # opt: 
        init_samples: List[List[int]] = []
        init_accs: List[float] = []
        
        query_setups: List[QuerySetup] = [] 
        # init BO 
        # 1. Random sample configs
        init_raw_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
        # for sample in init_samples:
        #     print(sample)
        # print(init_raw_samples)
        init_samples = []
        init_accs = []
        # 2. Run those config to get ACC & Latency
        for init_idx, sample in enumerate(init_raw_samples):
            sample[2] = init_idx % 5
            init_samples.append(sample) 

            config = sample_to_config(sample, query.knob_list)
            # print('init', config)
            result = self.simulate_pipeline(config=config, query=query)  
            setup_metric = self.postprocess_result(result, placement, config, query)
            init_accs.append(setup_metric.accuracy_index)

            query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))


        bo.fit(X=np.array(init_samples), y=np.array(init_accs))


        prune_count, stop_count = 0, 0  
        max_acc_index = max(init_accs)
        # continue
        while True:
            sample = bo.get_next_sample()
            config = sample_to_config(sample, query.knob_list) 
            # print('search', config)

            profile_result = self.simulate_pipeline(config=config, query=query)
            setup_metric = self.postprocess_result(profile_result, placement, config, query) 

            # Fit BO again
            init_samples.append(sample)
            init_accs.append(setup_metric.accuracy_index)
            bo.fit(X=np.array(init_samples), y=np.array(init_accs))

            query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))

            stop_count += 1 if setup_metric.accuracy_index < max_acc_index else 0

            max_acc_index = max(setup_metric.accuracy_index, max_acc_index)

            if  stop_count > self.bo_stop_count:
                break
        
        return query_setups
    
    def search_by_cbo(self, query: Query, cluster_state: ClusterState):
        """
        Search by Constraint BO
        """
        self.cache = {}
        self.profile_lat = 0

        placements: list[list] = generate_placement_choices()
        foot_print = []
        # total_profile_time = 0
        for placement in placements:
            obj_bo = BayesianOptimization(feat_bounds=self.feat_bounds, acquisition='ei') 
            acc_bo = BayesianOptimization(feat_bounds=self.feat_bounds)
            lat_bo = BayesianOptimization(feat_bounds=self.feat_bounds)


            init_samples = obj_bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
            init_gpu_times, init_lats, init_accs = [], [], []
            
            for sample in init_samples:
                
                # Convert BO sample back to config
                config = sample_to_config(sample, self.knob_list)
                result = self.simulate_pipeline(config=config)  

                post_result = self.postprocess_result(result, placement)
                init_gpu_times.append(post_result['gpu_time'])
                init_lats.append(post_result['latency'])
                init_accs.append(post_result['accuracy'])

                foot_print.append({
                    'gpu_time': post_result['gpu_time'],
                    'latency': post_result['latency'],
                    'accuracy': post_result['accuracy'],
                    'config': config,
                    'placement': placement,
                    'utility': post_result['utility'],
                    'type': 'random'
                })
                
            obj_bo.fit(X=init_samples, y=init_gpu_times)
            acc_bo.fit(X=init_samples, y=init_accs)
            lat_bo.fit(X=init_samples, y=init_lats)

            # BO optimization 
            prune_count, stop_count = 0, 0
            min_cost = min(init_gpu_times)

            while True:
                # Find next profile point
                samples = obj_bo._get_random_raw_samples()
                weighted_acq_vals = []
                for sample in samples:
                    acq_val = obj_bo._get_acq_val(sample)
                    pred_acc_mean, pred_acc_std = acc_bo.get_pred(sample)
                    pred_lat_mean, pred_lat_std = lat_bo.get_pred(sample)
                    # print(pred_lat_mean, pred_lat_std)
                    prob_meet_acc = 1 - probability_greater_than_t(pred_acc_mean, pred_acc_std, self.min_accuracy)
                    prob_meet_lat = 1 - probability_greater_than_t(pred_lat_mean, pred_lat_std, self.max_latency)
                    # print('----')
                    # print(pred_acc_mean, pred_acc_std, self.min_accuracy)
                    # print(pred_lat_mean, pred_lat_mean, self.max_latency)
                    # print(prob_meet_acc, prob_meet_lat)

                    w_acq_val = acq_val * prob_meet_acc * prob_meet_lat 
                    weighted_acq_vals.append(w_acq_val)
                    # print(sample, weighted_acq_vals, acq_val)
                next_sample = [s for _, s in sorted(zip(weighted_acq_vals, samples), reverse=True)][0]
                sample = next_sample
                # print(init_samples, sample)
                # exit(0)

                # print(next_sample)
                # print('init')
                # print(init_samples)
                # print(init_lats)
     
                # exit()
                
                # Simulate profilation 
                config = sample_to_config(sample, self.knob_list) 
                result = self.simulate_pipeline(config=config)
                post_result = self.postprocess_result(result, placement) 

                # Fit BO again
                init_samples.append(sample)
                init_gpu_times.append(post_result['gpu_time'])
                init_lats.append(post_result['latency'])
                init_accs.append(post_result['accuracy'])
                obj_bo.fit(X=init_samples, y=init_gpu_times)
                acc_bo.fit(X=init_samples, y=init_accs)
                lat_bo.fit(X=init_samples, y=init_lats)
                
                # print('-----')
                # print(post_result['gpu_time'], obj_bo.get_pred(sample))
                # print(post_result['accuracy'], acc_bo.get_pred(sample))
                # print(post_result['latency'], lat_bo.get_pred(sample))
                # print(init_samples)

                foot_print.append({
                    'gpu_time': post_result['gpu_time'],
                    'latency': post_result['latency'],
                    'accuracy': post_result['accuracy'],
                    'config': config,
                    'placement': placement,
                    'utility': post_result['utility'],
                    'type': 'search'
                })

                # stop criteria
                # prune_count += 1 if post_result['accuracy'] > self.min_accuracy or post_result['latency'] > self.max_latency else 0
                stop_count = (stop_count + 1) if post_result["gpu_time"] < min_cost else 0
                print(post_result['gpu_time'])

                min_cost = min(post_result["gpu_time"], min_cost)

                if prune_count >= self.early_prune_count or stop_count >= self.bo_stop_count:
                    break
            print('----> prune_cnt:{}, bo_cnt:{}'.format(prune_count, stop_count))            

        foot_print.sort(key=lambda x: x['gpu_time'], reverse=False) 
        # print(foot_print[:5])
        # exit()
        # filter:
        f_foot_print = []
        for d in foot_print:
            if d['latency'] < self.max_latency and d['accuracy'] < self.min_accuracy:
                f_foot_print.append(d)

        if len(f_foot_print) <= 0:
            print('----none----')
            return foot_print[0], self.profile_lat
        return f_foot_print[0], self.profile_lat
    
    def eval_setups(self, subtask_id, query_setups_per_query, score_key):
        first_query_setup = query_setups_per_query[0][subtask_id]    
        total_combinations = 1        
        for query_setups in query_setups_per_query[1:]:
            total_combinations *= len(query_setups) 

        best_combined_query_setup = None
        best_setup_score = ClusterState.INIT_SCORE 
        best_cluster_state = None

        st = time.time()
        for idx, partial_query_setup in enumerate(itertools.product(*query_setups_per_query[1:])):
            combined_query_setup = (first_query_setup,) + partial_query_setup
            cluster_state = ClusterState(self.cluster_spec)

            for query, query_setup in zip(self.queries, combined_query_setup):
                cluster_state.push_query(query, query_setup['placement'], query_setup)
            if (cluster_state.exceed_score(score_key, best_setup_score)):
                best_combined_query_setup = combined_query_setup
                best_cluster_state = cluster_state
                best_setup_score = cluster_state.get_score(score_key)

            if idx % 200000 == 0:
                print('subtask:', subtask_id, idx, "/", total_combinations, 'time', time.time() - st, flush=True)
                st = time.time()
                pass
        

        if best_cluster_state == None:
            return None
        else:
            return  {
                score_key: best_setup_score,
                'best_tot_util': best_cluster_state.tot_util, 
                'best_tot_gpu_time': best_cluster_state.tot_gpu_processing_time,
                'optimal_combined_query_setup': best_combined_query_setup,
            }

    def explore_all_query_setups(self, optimize_goal='utility'):
        query_setups_per_query = [] 
        num_process = len(self.queries)
        num_cpu = os.cpu_count()
        if num_cpu != None:
            num_process = min(num_process, num_cpu)

        with Pool(num_process) as pool:
            jobs = []
            for query in self.queries:
                if optimize_goal == 'utility':
                    job = pool.apply_async(self.search_by_utlity, (query, ClusterState(self.cluster_spec)))
                else:
                    job = pool.apply_async(self.search_by_accuracy, (query, ClusterState(self.cluster_spec))) 
                jobs.append(job)
            for job in jobs:
                query_setups = job.get()  
                query_setups_per_query.append(query_setups)

        # Filter low-accuracy setups
        valid_query_setups_per_query = []
        for idx, query_setups in enumerate(query_setups_per_query):
            query = self.queries[idx]
            valid_query_setups = []
            for query_setup in query_setups: 
                # print(query_setup)
                if query_setup['accuracy'] < query.acc_thold:
                    valid_query_setups.append(query_setup)
            valid_query_setups_per_query.append(valid_query_setups)

        print('Num searched query setups', [len(query_setups) for query_setups in query_setups_per_query])
        print('Num valid query setups', [len(query_setups) for query_setups in valid_query_setups_per_query])

        return valid_query_setups_per_query
        
    def exhaustive_setup_search(self):
        query_setups_per_query = self.explore_all_query_setups()

        num_process = len(self.queries)
        num_cpu = os.cpu_count()
        if num_cpu != None:
            num_process = min(num_process, num_cpu)
        
        score_key = 'utility'
         
        st = time.time()
        with Pool(num_process) as pool:
            jobs = []
            for subtask_id in range(len(query_setups_per_query[0])):
                job = pool.apply_async(self.eval_setups, (subtask_id, query_setups_per_query, score_key))
                jobs.append(job)

            # Wait for all jobs to finish
            results = []
            for job in jobs:
                result = job.get()
                if result != None:
                    results.append(result)
            
            sorted_result = sorted(results, key=lambda x: x[score_key], reverse=False)

            print("=====Result=====")
            # print([r['max_tot_util'] for r in sorted_result])
            print(sorted_result[0]['optimal_combined_query_setup'])
            # print(sorted_result[0]['optimal_cluster_state'].tier_state)
            print('Total Utility', sorted_result[0]['best_tot_util']) 
            print('Total GPU Time', sorted_result[0]['best_tot_gpu_time'])
        
        print('iterate time:', time.time()-st)                   

    def greedy_setup_search(self):
        query_setups_per_query = self.explore_all_query_setups()


        score_key = 'utility'
        
        num_queries = len(self.queries)
        for place_order in itertools.permutations([i for i in range(num_queries)], num_queries):
            print("=====PLACE=====") 
            cluster_state = ClusterState(self.cluster_spec)
            best_query_setups = [] #list of best query setup
            queries_meet_constraints = True
            for idx in place_order:
                query = self.queries[idx]
                query_setups = query_setups_per_query[idx]
                
                print('Place a query')
                best_query_setup = None 
                best_setup_score = 0 
                for query_setup in query_setups:
                    query_id = cluster_state.push_query(query, query_setup['placement'], query_setup)
                    if (cluster_state.exceed_score(score_key, best_setup_score)):
                        best_query_setup = query_setup 
                        best_setup_score = cluster_state.get_score(score_key) 
                
                    cluster_state.pop_query(query_id)

                if best_query_setup == None:
                    queries_meet_constraints = False
                    break
                best_query_setups.append(best_query_setup)
                cluster_state.push_query(query, best_query_setup['placement'], best_query_setup)

            print('======RESULT=====')
            if queries_meet_constraints: 
                # print(query_result)
                for query_setup in best_query_setups:
                    print(query_setup)
                # print(cluster_state.tier_state)
                print('Total Utiliy', cluster_state.tot_util)
                print('Total GPU Time', cluster_state.tot_gpu_processing_time)
            else:
                print('Can not find valid setups for all queries ><') 

    def beam_setup_search(self, num_beam=100):
        query_setups_per_query = self.explore_all_query_setups()

        cluster_state = ClusterState(self.cluster_spec)
        beams = [([], 0)] # (query_setup, tot_gpu_processing_time)
        for query, query_setups in zip(self.queries, query_setups_per_query):
            new_beams = []
            for query_setup_seq, _ in beams:
                for query_setup in query_setups:
                    new_query_setup_seq = query_setup_seq + [query_setup] 
                    # evaluate
                    cluster_state = ClusterState(self.cluster_spec) 
                    query_ids = []
                    for selected_query, selected_query_setup in zip(self.queries, new_query_setup_seq):
                        query_id = cluster_state.push_query(selected_query, selected_query_setup['placement'], selected_query_setup)
                        query_ids.append(query_id)
                    tot_gpu_processing_time = cluster_state.tot_gpu_processing_time
                    if (cluster_state.is_cluster_out_of_memory() == False
                        and cluster_state.queries_meet_latencies()):
                        new_beams.append((new_query_setup_seq, tot_gpu_processing_time))
            
            beams = sorted(new_beams, key=lambda x:x[1], reverse=False)[:min(len(beams), num_beam)]
        print('=====RESULT=====')
        if len(beams) == 0:
            print('Cant not find ><')
        else:
            print(beams[0][0])
            print('Total GPU Time', beams[0][1])

    def shrink_setup_search(self):
        self.num_init_sample_configs = 100 
        
        query_configs_by_query = self.explore_all_query_setups(optimize_goal='accuracy')
        
        print("=====Start Placing=====")
        while True:
             
            # Find the job with minimum accuracy but available accuracy. 
            # Find the low-latency placement for each job. 
            
            current_configs = []
            for idx, configs in enumerate(query_configs_by_query):
                current_configs.append(configs[0])
             

            print('=====Remaining config=====', [len(cs) for cs in query_configs_by_query])

            current_config_diff_placements_result_by_query = []
            for query_config, query in zip(current_configs, self.queries):
                config = query_config['config']
                placements = self.generate_placement_choices(query, self.cluster_spec )
                post_results = []
                for placement in placements:
                    profile_result = self.simulate_pipeline(config, query)
                    post_result = self.postprocess_result(profile_result, placement, config, query)
                    post_results.append(post_result)
                post_results.sort(key=lambda x: x['latency'])
                
                current_config_diff_placements_result_by_query.append(post_results) 
            
            best_queries_post_result = []
            for idx, config_placement_post_results in enumerate(current_config_diff_placements_result_by_query):
                best_queries_post_result.append(config_placement_post_results[0])
                current_config_diff_placements_result_by_query[idx].pop(0)
            # Step2 
            good = False
            while True:
                cluster_state = ClusterState(self.cluster_spec)
                query_ids = []
                for query, config_post_result in zip(self.queries, best_queries_post_result): 
                    query_id = cluster_state.push_query(query, config_post_result['placement'], config_post_result)
                    query_ids.append(query_id)

                some_query_violate_latency = False
                current_latencies = []
                current_accuracies = []
                for query, query_id in zip(self.queries, query_ids):
                    updated_post_result = cluster_state.get_query_profile(query_id)
                    if updated_post_result['latency'] >= query.lat_thold + 0.1:
                        some_query_violate_latency = True
                    current_latencies.append(updated_post_result['latency']) 
                    current_accuracies.append(updated_post_result['accuracy'])
                print('Cur ACC:', current_accuracies, 'LAT:', current_latencies) 

                if some_query_violate_latency or cluster_state.is_cluster_out_of_memory():
                    # print('Remaining placements:', [len(cc) for cc in current_config_diff_placements_result_by_query])
                    # print('OOM:',cluster_state.is_cluster_out_of_memory())
                    # print('Lateny violation:', some_query_violate_latency)
                    # find loose latency, than update. 
                    loose_idx = -1
                    loose = -100000000
                    for idx, query  in enumerate(self.queries):
                        if len(current_config_diff_placements_result_by_query[idx]) <= 0:
                            continue
                        query_loose = query.lat_thold - current_config_diff_placements_result_by_query[idx][0]['latency']  
                        # print('query loose', query_loose)
                        if query_loose > loose:
                            loose = query_loose 
                            loose_idx = idx

                    print('Loose:', loose)
                    if loose_idx == -1: 
                        break
                    else: 
                        best_queries_post_result[loose_idx] = current_config_diff_placements_result_by_query[loose_idx][0]
                        current_config_diff_placements_result_by_query[loose_idx].pop(0)
                else: 
                    good = True
                    break
             
            if good == True:
                print('=========RESULT================')
                # print(best_queries_post_result)
                cluster_state = ClusterState(self.cluster_spec)
                query_ids = []
                for query, best_query_post_result in zip(self.queries, best_queries_post_result):
                    query_id = cluster_state.push_query(query, best_query_post_result['placement'], best_query_post_result)
                    query_ids.append(query_id)
                for query_id in query_ids:
                    print(cluster_state.get_query_profile((query_id)))
                print('tot_utils: ', cluster_state.tot_util)
                print('tot_gpu_time', cluster_state.tot_gpu_processing_time)
                print("remaining config", [len(c) for c in query_configs_by_query])
                break
            else:
                #step 3
                best_gpu_diff = 10000000
                upgrade_config_idx = -1
                for idx, query in enumerate(self.queries):
                    if len(query_configs_by_query[idx]) <= 1:
                        continue
                    gpu_diff = query_configs_by_query[idx][1]['tot_gpu_time'] - current_configs[idx]['tot_gpu_time']
                    if gpu_diff < best_gpu_diff:
                        best_gpu_diff = gpu_diff
                        upgrade_config_idx = idx
                if upgrade_config_idx == -1:
                    print("============GIVE UP=========")
                    print("Cant not find")
                    break
                else:
                    query_configs_by_query[upgrade_config_idx].pop(0)
                                        
                    # print('remaining configs:', [len(cs) for cs in query_configs_by_query])
                   
    def search_by_utility_multi_queries(self, subtask_id, num_subtask,  placement_by_query):
        # print('Subtask', subtask_id, '/', num_subtask, 'start ...')
        st = time.time()
        def simulate_sample(sample: List[int]):
            sample_sid = 0
            cluster_state = ClusterState(self.cluster_spec)
            query_ids = []
            # st = time.time()
            for query, placement in zip(self.queries, placement_by_query):
                st = time.time() 
                sample_per_query = sample[sample_sid:sample_sid+len(query.knob_list)]        
                sample_sid += len(query.knob_list)
                # print('get sample', time.time() - st)
                # st = time.time()
                config_per_query = sample_to_config(sample_per_query, query.knob_list)
                # print('sample 2 config', time.time() - st)
                # st = time.time() 
                result_per_query = self.simulate_pipeline(config_per_query, query)
                # print('simulate', time.time() - st)
                # st = time.time()
                post_result_per_query = self.postprocess_result(result_per_query, placement, config_per_query, query)
                # print('post process', time.time() - st)
                # st = time.time()
                query_id = cluster_state.push_query(query, placement, post_result_per_query)
                # print('push', time.time() - st)

                query_ids.append(query_id)
            # print('Sample BO', time.time() - st)
            # st = time.time()
            # After placing 
            deployed_profiles = []
            for query_id in query_ids:
                deployed_profile = cluster_state.get_query_profile(query_id)
                deployed_profiles.append(deployed_profile) 
            # print('Sample place', time.time() - st)
            global_query_setup = {
                'placement': placement_by_query,
                'profile': deployed_profiles, 
                'cluster_state': cluster_state.tier_state,
                'tot_util': cluster_state.tot_util, 
                # 'type': 'random'
            }
            return cluster_state, global_query_setup 

        
        feat_bounds = []
        for query in self.queries:
            feat_bounds_per_query = self.get_feat_bounds(query)
            feat_bounds += feat_bounds_per_query
        
        bo = BayesianOptimization(feat_bounds=feat_bounds)
         
        init_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
        init_utils = []
        num_exploded_cluster = 0
        best_tot_util = -10000
        best_global_query_setup = None
        # print('get new', time.time() - st)        
        # st = time.time()
        for sample in init_samples:
            cluster_state, global_query_setup = simulate_sample(sample)
            # print('sample (sim)', time.time() - st) 
            init_utils.append(cluster_state.tot_util)
            if cluster_state.is_cluster_out_of_memory():
                num_exploded_cluster += 1
            else:
                if cluster_state.tot_util > best_tot_util:
                    best_tot_util = cluster_state.tot_util
                    best_global_query_setup = global_query_setup 

            # print('sample', time.time() - st)
            # st = time.time()
        if num_exploded_cluster == self.num_init_sample_configs:
            print('subtask', subtask_id, '/', num_subtask, 'done', time.time() - st, flush=True)
            return best_tot_util, None
            
        bo.fit(X=np.array(init_samples), y=np.array(init_utils))
        # Start searching
        early_prune_cnt, bo_stop_cnt = 0, 0
        max_tot_util = -100000000
        while True: 
            sample = bo.get_next_sample()                 
            cluster_state, global_query_setup = simulate_sample(sample)
            tot_util = cluster_state.tot_util
            if cluster_state.is_cluster_out_of_memory() == False:
                if  tot_util > best_tot_util:
                    best_global_query_setup = global_query_setup
                    best_tot_util = tot_util
            
            #early prune or bo stop 
            early_prune_cnt += 1 if tot_util < 0 else 0
            bo_stop_cnt = bo_stop_cnt + 1 if tot_util <= max_tot_util else 0 
            max_tot_util = max(tot_util, max_tot_util) 

            if early_prune_cnt >= self.early_prune_count:
                break 
            if bo_stop_cnt >= self.bo_stop_count:
                break
            # print(early_prune_cnt, bo_stop_cnt, max_tot_util)

        print('subtask', subtask_id, '/', num_subtask, 'done', time.time() - st, flush=True)
        return best_tot_util, best_global_query_setup

        self.num_init_sample_configs = self.num_init_sample_configs ** len(self.queries)
        self.early_prune_count = self.num_init_sample_configs * len(self.queries)
        
        plcs_by_query = [self.generate_placement_choices(query, self.cluster_spec) for query in self.queries]        

        total_combinations = 1
        for lst in plcs_by_query:
            total_combinations *= len(lst)

        best_global_query_setup = None
        best_tot_util = -10000
        with Pool(4) as pool:
            jobs = [] 
            for subtask_id, plcs in enumerate(itertools.product(*plcs_by_query)):
                job = pool.apply_async(self.search_by_utility_multi_queries, (subtask_id, total_combinations, plcs))
                jobs.append(job)
            for job in jobs:
                tot_util, global_query_setup = job.get()
                if global_query_setup != None and tot_util > best_tot_util:
                    best_tot_util = tot_util
                    best_global_query_setup = global_query_setup

        # for subtask_id, plcs in tqdm(enumerate(itertools.product(*plcs_by_query))):
        #     tot_util, global_query_setup = self.vulcan_search_multiple_query(subtask_id, total_combinations, plcs)
        #     if global_query_setup != None and tot_util > best_tot_util:
        #         best_tot_util = tot_util
        #         best_global_query_setup = global_query_setup          

        print(best_global_query_setup)

        pass
  
    def joint_optimize(self):
        self.num_init_sample_configs = self.num_init_sample_configs ** len(self.queries)
        self.early_prune_count = self.num_init_sample_configs * len(self.queries)
        
        plcs_by_query = [gen_all_place_choices(query, self.cluster_spec) for query in self.queries]        

        total_combinations = 1
        for lst in plcs_by_query:
            total_combinations *= len(lst)

        best_global_query_setup = None
        best_tot_util = -10000
        with Pool(4) as pool:
            jobs = [] 
            for subtask_id, plcs in enumerate(itertools.product(*plcs_by_query)):
                job = pool.apply_async(self.search_by_utility_multi_queries, (subtask_id, total_combinations, plcs))
                jobs.append(job)
            for job in jobs:
                tot_util, global_query_setup = job.get()
                if global_query_setup != None and tot_util > best_tot_util:
                    best_tot_util = tot_util
                    best_global_query_setup = global_query_setup

        

        print(best_global_query_setup)


    
if __name__ == "__main__":
    # queries
    queries = [
        Query(0.35, 0.3, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg), 
        Query(0.45, 0.2, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        Query(0.8, 0.1, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        Query(0.5, 0.2, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        Query(0.5, 0.4, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
    ]

    # Cluster Spec
    mach_info = {
        "compute": [0.1, 0.22, 1],
        "bandwidth": [0.08, 0.05, 0] # bw=(t0->t1, t1->t2, t2->t3)
    }
    cluster_spec = [
        TierSpec('edge', num_nodes=1, num_gpus=1, gpu_memory_size=1000, compute_speed=mach_info['compute'][0], bandwidth=mach_info['bandwidth'][0]),
        TierSpec('on-perm', num_nodes=1, num_gpus=1, gpu_memory_size=2000, compute_speed=mach_info['compute'][1], bandwidth=mach_info['bandwidth'][1]),
        TierSpec('cloud', num_nodes=1, num_gpus=1, gpu_memory_size=3000, compute_speed=mach_info['compute'][2], bandwidth=mach_info['bandwidth'][2]),
    ]



    # Search single query
    engine = Engine(queries=[], cluster_spec=cluster_spec, num_profile_sample=5000)
    query_setups = engine.search_by_utlity(queries[0], ClusterState(cluster_spec))
    query_setups = sorted(query_setups, key = lambda x: x.setup_metric.utility, reverse=True);
    print(query_setups[0])

    exit()
    ## Search multi-query
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['shrink', 'greedy', 'exhaustive', 'beam', 'joint'], required=True)
    parser.add_argument('-q', '--num_queries', required=True)
    args = parser.parse_args()
    



    # Filter queries to conduct experiment on different #queries.
    queries = queries[:int(args.num_queries)] 
    engine = Engine(queries, cluster_spec, num_profile_sample=5000)

    
    """
    Goal: Let every query satisfy "latency & accuracy" constraints while minimizing gpu processing time. 
    
    Method:
        - Disaggregate Searching & Placing (...search)
        - Combine Searching & Placing (...optimize)
    """
    start_time = time.process_time()
    st = time.time()
    if args.method == 'joint':
        engine.joint_optimize()
    elif args.method == 'greedy':
        engine.greedy_setup_search()
    elif args.method == 'exhaustive':
        engine.exhaustive_setup_search()
    elif args.method == 'beam':
        engine.beam_setup_search()
    elif args.method == 'shrink':
        engine.shrink_setup_search()

    print("======PROFILE=======")
    print('total time:', time.time() - st)
    print('tota cpu processing time', time.process_time() - start_time)
