from bo import BayesianOptimization
import docker 
import socket
import json 
import copy 
from task import speech_recognition_ops, speech_recongnition_knobs, Knob, Operator, speech_recognition_dnn_usg
from typing import Tuple, Dict, List
from scipy.stats import norm
from dataclasses import dataclass
import numpy as np
import itertools
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



mach_info = {
    "compute": [0.1, 0.22, 1],
    "bandwidth": [0.08, 0.05, 0] #gbs 
}




def probability_greater_than_t(mu, sigma, t):
    z = (t - mu) / sigma
    cdf_value = norm.cdf(z)
    probability = 1 - cdf_value
    return probability

def sample_to_config(sample: List[int], knob_list: List[Knob])-> dict: 
    config = {}
    assert len(sample) == len(knob_list), "Sample and should have equal size as knob"

    for knob_idx, knob in enumerate(knob_list): 
        config[knob.name] = knob.choice[sample[knob_idx]]
    return config

def config_to_sample(config: dict, knob_list: List[Knob]) -> List[int]:
    sample = []
    for knob in knob_list:  
        i = knob.choice.index(config[knob.name])
        sample.append(i)
    return sample

@dataclass
class Query:
    acc_thold: float 
    lat_thold: float
    profile_result_file: str
    knob_list: List[Knob]
    op_list: List[Operator]
    dnn_gpu_usg: Dict[str, int]
@dataclass
class TierSpec:
    name: str
    num_nodes: int
    num_gpus: int
    gpu_memory_size: int
    compute_speed: float
    bandwidth: float
@dataclass
class NodeState:
    remain_gpu_memory: float
    num_gpu_ps: int = 0
    # ps: List[Operator] = []
    # ps_gpu_times: List[int] = []

    

class ClusterState:
    def __init__(self, spec: List[TierSpec]):
        # op, ps, gpu_time, memory
        self.next_query_uuid = 0

        self.spec = spec
        self.tier_state: List[List[NodeState]] = [] 

        self.deployed_queries = {}
        for tier_spec in spec:
            node_states = [NodeState(tier_spec.gpu_memory_size) for _ in range(tier_spec.num_nodes)] 
            self.tier_state.append(node_states)

        self.tot_util = 0
            
    def _find_node_level_plc(self, query, plc: List[int]): 
        # place them on node 0. 
        node_level_plc = []
        for tier_id in plc:
            node_level_plc.append({
                'tier_id': tier_id,
                'node_id': 0 
            })
            
        return node_level_plc 

    def _add_query(self, tier_level_plc: List[int], query: Query, profile_result: dict):
        node_level_plc = self._find_node_level_plc(query, tier_level_plc)

        for plc_by_op, op, gpu_mem_usg in zip(node_level_plc, query.op_list, profile_result['gpu_mem_usgs']):
            # print(tier_id, op.tpy, gpu_mem_usg)
            tier_id = plc_by_op['tier_id']
            node_id = plc_by_op['node_id']
            if op.tpy == 'dnn':
                self.tier_state[tier_id][node_id].num_gpu_ps += 1
                self.tier_state[tier_id][node_id].remain_gpu_memory -= gpu_mem_usg

        # place query
        self.deployed_queries[self.next_query_uuid] = {
            'query': query,
            'placement': tier_level_plc,
            'node_level_plc': node_level_plc,
            'profile_result': profile_result
        } 

        query_uuid = self.next_query_uuid 

        self.next_query_uuid += 1

        return query_uuid 
    
    def _remove_query(self, query_id):
        node_level_plc = self.deployed_queries[query_id]['node_level_plc']
        query = self.deployed_queries[query_id]['query']
        profile_result = self.deployed_queries[query_id]['profile_result']    

        for plc_by_op, op, gpu_mem_usg in zip(node_level_plc, query.op_list, profile_result['gpu_mem_usgs']):
            # print(tier_id, op.tpy, gpu_mem_usg)
            tier_id = plc_by_op['tier_id']
            node_id = plc_by_op['node_id']
            if op.tpy == 'dnn':
                self.tier_state[tier_id][node_id].num_gpu_ps -= 1
                self.tier_state[tier_id][node_id].remain_gpu_memory += gpu_mem_usg
        self.deployed_queries.pop(query_id)        

    def is_cluster_out_of_memory(self):
        for tier_id in range(0, len(self.tier_state)):
            for node_id in range(0, len(self.tier_state[tier_id])):
                if self.tier_state[tier_id][node_id].remain_gpu_memory < 0: 
                    return True
        return False

    def _recompute_global_profile(self):

        self.tot_util = 0
        self.tot_gpu_processing_time = 0

        # update latency, update utility
        for query_id, deployed_query in self.deployed_queries.items():
            compute_latency = 0
            profile_result = deployed_query['profile_result']
            for op, tier_id, op_latency in zip(deployed_query['query'].op_list, deployed_query['placement'], profile_result['compute_latency_by_op']):
                if op.tpy == 'dnn': 
                    # GPU bottleneck
                    compute_latency += op_latency * self.tier_state[tier_id][0].num_gpu_ps
                else: 
                    compute_latency += op_latency

            pipe_latency = compute_latency + profile_result['transfer_latency']
            utility = (0.5 * -1 * (profile_result['accuracy'] - deployed_query['query'].acc_thold) + 0.5 * (deployed_query['query'].lat_thold - (pipe_latency))) / profile_result['tot_gpu_time'] 
            deployed_query['profile_result']['utility'] = utility
            
            deployed_query['profile_result']['latency'] = pipe_latency 

            self.deployed_queries[query_id] = deployed_query 

            # global metric
            self.tot_util += utility
            self.tot_gpu_processing_time += deployed_query['profile_result']['gpu_time']

    def get_query_profile(self, query_id):
        return self.deployed_queries[query_id]['profile_result']
    
    def push_query(self, query: Query, tier_level_plc: List[int], profile_result):
        query_id = self._add_query(tier_level_plc, query, profile_result)
        self._recompute_global_profile()
        return query_id
        
    def pop_query(self, query_id):
        self._remove_query(query_id)
        self._recompute_global_profile() 
        return True

class Engine:
    def __init__(self, num_profile_sample=-1):
        self.early_prune_count = 10 
        self.bo_stop_count = 3
        self.num_init_sample_configs = 5
        self.utility_thold = 0  

        # self.min_accuracy = min_accuracy
        # self.max_latency = max_latency
        self.num_profile_sample = num_profile_sample

        self.queries = [
            Query(0.35, 0.3, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg), 
            Query(0.45, 0.2, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
            Query(0.8, 0.05, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
            Query(0.5, 0.2, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
            # Query(0.5, 0.4, "speech_recognition/profile_result_1.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        ]

        self.cache = {}

        # self.feat_bounds = self.get_feat_bounds()

        self.profile_lat = 0        

        self.cluster_spec = [
            TierSpec('edge', num_nodes=1, num_gpus=1, gpu_memory_size=1000, compute_speed=mach_info['compute'][0], bandwidth=mach_info['bandwidth'][0]),
            TierSpec('on-perm', num_nodes=1, num_gpus=1, gpu_memory_size=2000, compute_speed=mach_info['compute'][1], bandwidth=mach_info['bandwidth'][1]),
            TierSpec('cloud', num_nodes=1, num_gpus=1, gpu_memory_size=3000, compute_speed=mach_info['compute'][2], bandwidth=mach_info['bandwidth'][2]),
        ]

    def get_feat_bounds(self, query: Query):
        feat_bounds = []
        for knob in query.knob_list: 
            feat_bounds.append([i for i in range(len(knob.choice))]) 
        
        return feat_bounds

    def simulate_run_pipeline(self, config: dict, query: Query) -> dict:

        config_str = json.dumps(config)
        # print(config_str)
        if config_str in self.cache:
            return self.cache[config_str]

        with open (query.profile_result_file, 'r') as fp:
            profile_result = json.load(fp) 
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

    def postprocess_result(self, profile_result: dict, placement: List[int], config: dict, query: Query):
        # Vulcan implmentation: 
        # Uq,p,c = Pq,p,c/Rq,p,c
        # Pq,p,c(A,L) = g·aA ·(A - Am)+(1 - g)·aL ·(Lm - L)
        #             = weight * Accuracy + weight * Latency
        # Rq,p,c = agpu · Rgpu + anet · Rnet
        #        = gpu processing time + consumed network bandwidth
        
        # perf = perf_weight * exec_stat.accuracy + (1 - perf_weight) * exec_stat.latency 
        # resource = compute_coef * exec_stat.compute_cost + network_coef * exec_stat.network_cost
        # utility = perf / resource
        # return utility 

        # assert len(self.ops) == len(placement)

        prev_op_machine = 0
        tot_compute_latency = 0
        tot_transfer_latency = 0
        tot_gpu_time = 0
        compute_latency_by_op = []
        gpu_mem_usgs = []
        for op, op_machine in zip(query.op_list, placement):
            op_name = op.name
            op_type = op.tpy

            compute_latency = profile_result[f"{op_name}_compute_latency"] / mach_info['compute'][op_machine]
            compute_latency_by_op.append(compute_latency)
            # TODO: Ensure this assumption is correct:
            # I assume that compute_latency would increase when multiple user use same GPU. 
            if op.tpy == 'dnn':
                # compute_latency = cluster_state.get_latency_after_placing(op_machine, op, compute_latency)
                gpu_mem_usgs.append(query.dnn_gpu_usg[config['model']])
            else:
                gpu_mem_usgs.append(0)

            transfer_latency = 0
            for machine_id in range(prev_op_machine, op_machine):
                bandwidth = mach_info["bandwidth"][machine_id] #gbs
                bandwidth *= 10e9 #bytes/s
                transfer_latency += profile_result[f"{op_name}_input_size"] / bandwidth
            
            tot_compute_latency += compute_latency
            tot_transfer_latency += transfer_latency
            if op_type == 'dnn':
                tot_gpu_time += profile_result[f"{op_name}_compute_latency"] 

        pipe_accuracy = profile_result["cummulative_accuracy"][self.num_profile_sample - 1]
        pipe_latency = tot_compute_latency + tot_transfer_latency

        utility = (0.5 * -1 * (pipe_accuracy - query.acc_thold) + 0.5 * (query.lat_thold - pipe_latency)) / tot_gpu_time

        # print(pipe_latency, tot_compute_latency, tot_transfer_latency)
        # exit()
        post_result = {
            "utility": utility,
            "gpu_time": tot_gpu_time * 1000,
            "accuracy": pipe_accuracy,
            "latency": pipe_latency, 
            "compute_latency_by_op": compute_latency_by_op, 
            "transfer_latency": tot_transfer_latency, 
            "config": config, 
            "placement": placement,
            "gpu_mem_usgs": gpu_mem_usgs,
            "tot_gpu_time": tot_gpu_time
        }
       
        return post_result
    
    def gen_all_place_choices(self, query: Query, cluster_spec: List[TierSpec]):
        
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

    def vulcan_search(self, query: Query, cluster_state: ClusterState, optimize: bool=False):
        """
        Learn a BO  
        """
        self.cache = {}
        self.profile_lat = 0
        feat_bounds = self.get_feat_bounds(query) 
        placements: list[list] = self.gen_all_place_choices(query, cluster_state.spec)

        foot_print = []
        # total_profile_time = 0
        for placement in placements:
            bo = BayesianOptimization(feat_bounds=feat_bounds) 
            init_utils = [] # utility

            # opt: 
            init_samples: List[List[int]] = []
            init_utils: List[float] = []
            
            if optimize and len(self.cache) >= 3 :
                for config_str, result in self.cache.items():
                    config = json.loads(config_str)
                    post_result = self.postprocess_result(result, placement, config, query)
                    sample = config_to_sample(config, query.knob_list) 
                    #
                    init_samples.append(sample)
                    init_utils.append(post_result['utility'])
                    # utilities.append((utility, appox_latency, result["accuracy"], config, placement, 'prev'))

                bo.fit(X=np.array(init_samples), y=np.array(init_utils))
            else: 
                
                # init BO 
                # 1. Random sample configs
                init_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
                init_utils = []
                # 2. Run those config to get ACC & Latency
                for sample in init_samples:
                    
                    config = sample_to_config(sample, query.knob_list)
                    result = self.simulate_run_pipeline(config=config, query=query)  

                    post_result = self.postprocess_result(result, placement, config, query)
                     # extend
                    # post_result['config'] = config
                    # post_result['placement'] = placement
                    post_result['type'] = 'random'

                    init_utils.append(post_result['utility'])

                    foot_print.append(post_result)

                bo.fit(X=np.array(init_samples), y=np.array(init_utils))


            prune_count, stop_count = 0, 0  
            max_util = max(init_utils)
            # continue
            while True:
                sample = bo.get_next_sample()
                config = sample_to_config(sample, query.knob_list) 

                result = self.simulate_run_pipeline(config=config, query=query)
                post_result = self.postprocess_result(result, placement, config, query) 

                # extend
                # post_result['config'] = config
                # post_result['placement'] = placement
                post_result['type'] = 'search'

                # Fit BO again
                init_samples.append(sample)
                init_utils.append(post_result['utility'])
                bo.fit(X=np.array(init_samples), y=np.array(init_utils))

                foot_print.append(post_result)

                prune_count += 1 if post_result['utility'] < self.utility_thold else 0
                stop_count += 1 if post_result["utility"] < max_util else 0

                max_util = max(post_result["utility"], max_util)

                if prune_count >= self.early_prune_count or stop_count > self.bo_stop_count:
                    break
            # print(foot_print[len(foot_print) - 1])
        #     print(placement, prune_count, stop_count)
        # exit()
        
        return foot_print 

    def constraint_bo(self):
        self.cache = {}
        self.profile_lat = 0

        placements: list[list] = self.gen_all_place_choices()
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
                result = self.simulate_run_pipeline(config=config)  

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
                result = self.simulate_run_pipeline(config=config)
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

    def exhuastive_optimize(self):
        
        all_query_setups = [] 
        for query in self.queries:
            per_query_setups = self.vulcan_search(query, ClusterState(self.cluster_spec))
            
            valid_per_query_setups = []
            for per_query_setup in per_query_setups:
                if per_query_setup['accuracy'] < query.acc_thold and per_query_setup['latency'] < query.lat_thold:
                    valid_per_query_setups.append(per_query_setup)    
            all_query_setups.append(per_query_setups) 

        print(len(all_query_setups), [len(per_query_setups) for per_query_setups in all_query_setups])
        total_combintation = 1
        for per_query_setups in all_query_setups:
            total_combintation *= len(per_query_setups) 

        max_tot_util = -100000
        optimal_all_query_setup = None
        optimal_cluster_state = ClusterState(self.cluster_spec)
        for all_query_setup in tqdm(itertools.product(*all_query_setups), total=total_combintation):
            cluster_state = ClusterState(self.cluster_spec)
            for query, per_query_setup in zip(self.queries, all_query_setup):
                cluster_state.push_query(query, per_query_setup['placement'], per_query_setup)
            # print(cluster_state.tier_state)
            if cluster_state.is_cluster_out_of_memory() == False and cluster_state.tot_util > max_tot_util:
                max_tot_util = cluster_state.tot_util
                optimal_all_query_setup = all_query_setup
                optimal_cluster_state = cluster_state 

        print(optimal_all_query_setup) 
        print(optimal_cluster_state.tier_state)
        print(max_tot_util)
                   

    def single_optimize(self):
        cluster_state = ClusterState(self.cluster_spec)
        query_result = []
        for query in self.queries:
            foot_print = self.vulcan_search(query, ClusterState(self.cluster_spec))
            
            #TODO: Remove this assumtion that there must be a placement that is qualify
            # print(foot_print)
            placable_footprint = []
            cluster_utils = []
            for d in foot_print:
                plc = d['placement']

                query_id = cluster_state.push_query(query, plc, profile_result=d)
                if cluster_state.is_cluster_out_of_memory() == False:
                    placable_footprint.append(d)
                    cluster_utils.append(cluster_state.tot_util) 
                cluster_state.pop_query(query_id)
            print(len(placable_footprint)) 

            #sort by total utility
            sorted_footprint = [d for _, d in sorted(zip(cluster_utils, placable_footprint), key=lambda x: x[0], reverse=True)]

            query_result.append(sorted_footprint[0])
            cluster_state.push_query(query, sorted_footprint[0]['placement'], sorted_footprint[0])
            print(len(cluster_state.deployed_queries))

        print('===========')
        # print(query_result)
        for qr in query_result:
            print(qr)
        print(cluster_state.tier_state)
        print('tot_gpu_processing_time', cluster_state.tot_gpu_processing_time)
        print('tot_util', cluster_state.tot_util)

    def joint_optimize(self):
        self.num_init_sample_configs = self.num_init_sample_configs ** len(self.queries)
        self.early_prune_count = self.num_init_sample_configs * len(self.queries)
        


        plcs_by_query = [self.gen_all_place_choices(query, self.cluster_spec) for query in self.queries]        
        print(plcs_by_query)

        foot_print = [] 

        total_combinations = 1
        for lst in plcs_by_query:
            total_combinations *= len(lst)
        
        for plcs in tqdm(itertools.product(*plcs_by_query), total=total_combinations):
            feat_bounds = []
            for query in self.queries:
                feat_bounds_per_query = self.get_feat_bounds(query)
                # print(feat_bounds_per_query)
                feat_bounds += feat_bounds_per_query
            
            bo = BayesianOptimization(feat_bounds=feat_bounds)
            
            
            init_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
            init_utils = []
            num_exploded_cluster = 0

            for sample in init_samples:
                # Simulate each pipeline:
                # old_tot_util = 0
                sample_sid = 0
                cluster_state = ClusterState(self.cluster_spec)
                query_ids = []
                for query, plc in zip(self.queries, plcs):
                    sample_per_query = sample[sample_sid:sample_sid+len(query.knob_list)]        
                    sample_sid += len(query.knob_list)
                    # print(sample_per_query)
                    config_per_query = sample_to_config(sample_per_query, query.knob_list)
                    # print(config_per_query)
                    result_per_query = self.simulate_run_pipeline(config_per_query, query)
                    # print(result_per_query)
                    post_result_per_query = self.postprocess_result(result_per_query, plc, config_per_query, query)

                    uuid = cluster_state.push_query(query, plc, post_result_per_query)

                    # print(post_result_per_query)
                    # print()
                    # old_tot_util += post_result_per_query['utility']

                    query_ids.append(uuid)
                # print('======== After placing ==========')
                tot_util = 0 
                deployed_profiles = []
                for query_id in query_ids:
                    deployed_profile = cluster_state.get_query_profile(query_id)
                    deployed_profiles.append(deployed_profile) 
                    # print()
                    tot_util += deployed_profile['utility'] 
                     

                if cluster_state.is_cluster_out_of_memory() == True:
                    num_exploded_cluster += 1
                else: 
                    foot_print.append({
                        'placement': plcs,
                        'profile': deployed_profiles, 
                        'cluster_state': cluster_state.tier_state,
                        'tot_util': tot_util, 
                        'type': 'random'
                    })  
                    # exit()

                # print(old_tot_util, 'new:', tot_util)
                # print(cluster_state.tier_state, 'explode:', cluster_state.exploded)

                init_utils.append(tot_util)

                # foot_print.append(post_result)
            # :('num exploded', num_exploded_cluster)            

            if num_exploded_cluster == self.num_init_sample_configs:
                # print('Go to next placement')
                continue
                
            bo.fit(X=np.array(init_samples), y=np.array(init_utils))
            # Start searching
            early_prune_cnt, bo_stop_cnt = 0, 0
            max_tot_util = -100000000
            while True: 
                sample = bo.get_next_sample()                 
                 # Simulate each pipeline:
                # old_tot_util = 0
                sample_sid = 0
                cluster_state = ClusterState(self.cluster_spec)
                query_ids = []
                for query, plc in zip(self.queries, plcs):
                    sample_per_query = sample[sample_sid:sample_sid+len(query.knob_list)]        
                    sample_sid += len(query.knob_list)
                    # print(sample_per_query)
                    config_per_query = sample_to_config(sample_per_query, query.knob_list)
                    # print(config_per_query)
                    result_per_query = self.simulate_run_pipeline(config_per_query, query)
                    # print(result_per_query)
                    post_result_per_query = self.postprocess_result(result_per_query, plc, config_per_query, query)

                    uuid = cluster_state.push_query(query, plc, post_result_per_query)

                    query_ids.append(uuid)
                    # print(post_result_per_query)
                    # print()
                    # old_tot_util += post_result_per_query['utility']
                # print('query ids:', query_ids)
                deployed_profiles = []
                for query_id in query_ids:
                    deployed_profile = cluster_state.get_query_profile(query_id)
                    # print(deployed_profile)
                    deployed_profiles.append(deployed_profile) 
                tot_util = cluster_state.tot_util

                if cluster_state.is_cluster_out_of_memory == False:
                    foot_print.append({
                        'placement': plcs,
                        'profile': deployed_profiles, 
                        'cluster_state': cluster_state.tier_state,
                        'tot_util': tot_util, 
                        'type': 'search'
                    })

                #early prune or bo stop 
                early_prune_cnt += 1 if tot_util < 0 else 0
                bo_stop_cnt = bo_stop_cnt + 1 if tot_util <= max_tot_util else 0 
                max_tot_util = max(tot_util, max_tot_util) 

                if early_prune_cnt >= self.early_prune_count:
                    break 
                if bo_stop_cnt >= self.bo_stop_count:
                    break
                print(early_prune_cnt, bo_stop_cnt, max_tot_util)

        foot_print.sort(key=lambda x: x['tot_util'], reverse=True)
        print(foot_print[0])
        # print(len(foot_print))
        return foot_print 

    def exhausitive_search(self):
        poss_plc_per_query = []
        poss_config_per_query = []
        for query in self.queries:
            poss_plc = self.gen_all_place_choices(query, self.cluster_spec)
            poss_plc_per_query.append(poss_plc)

            poss_config = self.generate_all_configs(query)
            poss_config_per_query.append(poss_config)
        
        # print(poss_config_per_query)
        import itertools
        # result = 
        # print(result)
        i = 0 
        for plcs in itertools.product(*poss_plc_per_query):
            for configs in itertools.product(*poss_config_per_query):
                new_cluster_state = ClusterState(self.cluster_spec)
                post_results = []
                for plc, config, query in zip(plcs, configs, self.queries):
                    profile_result = self.simulate_run_pipeline(config, query)        
                    post_result = self.postprocess_result(profile_result, plc, query, cluster_state, config)
                    post_results.append(post_result) 

                # check placeable
                can_place = True
                for plc, config, query, post_result in zip(plcs, configs, self.queries, post_results):
                    if new_cluster_state.can_place_pipline(plc, query.op_list, post_result['gpu_mem_usgs']):
                        new_cluster_state.update(plc, post_result['gpu_mem_usgs'])
                    else:
                        can_place = False
                        break
                for plc, config, query, post_result in zip(plcs, configs, self.queries, post_results):
                    print(plc, config) 

                if can_place == False : continue

                # print(post_results)
                
                # update latency
                can_place = True
                for plc, config, query, post_result in zip(plcs, configs, self.queries, post_results):
                    print(plc, config) 
                
                
                # cal tot_gpu_time for further sorting




                exit()

        #     print(r)
        #     i += 1
        # print(i)

    def print_result(self):
        pass
        # print #qualify query, # tot_gpu_time(cloud gpu time)
   

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['single', 'joint', 'exhaustive'], required=True)
    args = parser.parse_args()
    engine = Engine(num_profile_sample=5000)
    if args.method == 'joint':
        engine.joint_optimize()
    elif args.method == 'single':
        engine.single_optimize()
    elif args.method == 'exhaustive':
        engine.exhuastive_optimize()
    # engine.single_optimize()


"""
{'placement': ([0, 1, 1, 2], [0, 1, 2, 2], [1, 1, 2, 2]), 'profile': 
[{'utility': 5.086042993358486, 'gpu_time': 2.0310568809509277, 'accuracy': 0.3484612275645886, 'latency': 0.09493792006595127, 'compute_latency_by_op': [0.3050851821899414, 0.05508780479431152, 0.09232076731595126, 0.09792938232421874], 'transfer_latency': 0.0026171527500000002, 'config': {'audio_sr': 14000, 'freq_mask': 4000, 'model': 'hubert-large'}, 'placement': [0, 1, 1, 2], 'gpu_mem_usgs': [0, 0, 2000, 0], 'tot_gpu_time': 0.020310568809509277}, 
{'utility': 6.027449276074536, 'gpu_time': 2.024819850921631, 'accuracy': 0.3616419270891887, 'latency': 0.04426809401843262, 'compute_latency_by_op': [0.2631664276123047, 0.05475185134194114, 0.02024819850921631, 0.09687254428863526], 'transfer_latency': 0.0037716970000000005, 'config': {'audio_sr': 14000, 'freq_mask': 500, 'model': 'hubert-large'}, 'placement': [0, 1, 2, 2], 'gpu_mem_usgs': [0, 0, 2000, 0], 'tot_gpu_time': 0.02024819850921631}, 
{'utility': 7.659651567409671, 'gpu_time': 1.975371837615967, 'accuracy': 0.35096379440343944, 'latency': 0.046423005752319335, 'compute_latency_by_op': [0.01598921689120206, 0.05630894140763716, 0.01975371837615967, 0.10268964767456054], 'transfer_latency': 0.006915569, 'config': {'audio_sr': 16000, 'freq_mask': 3000, 'model': 'hubert-large'}, 'placement': [1, 1, 2, 2], 'gpu_mem_usgs': [0, 0, 2000, 0], 'tot_gpu_time': 0.01975371837615967}], 'tot_util': 18.773143836842692}

{'utility': 6.4076665782014945, 'gpu_time': 1.9707894325256345, 'accuracy': 0.3496364547601433, 'latency': 0.047800313650512696, 'compute_latency_by_op': [0.013537190177223898, 0.012384700775146484, 0.019707894325256346, 0.0983954906463623], 'transfer_latency': 0.008384525, 'config': {'audio_sr': 16000, 'freq_mask': 4000, 'model': 'hubert-large'}, 'placement': [1, 2, 2, 2], 'gpu_mem_usgs': [0, 0, 2000, 0], 'tot_gpu_time': 0.019707894325256346, 'type': 'search'}
{'utility': 5.175026436054828, 'gpu_time': 1.975371837615967, 'accuracy': 0.35096379440343944, 'latency': 0.09458417598254394, 'compute_latency_by_op': [0.01598921689120206, 0.05630894140763716, 0.08978962898254395, 0.10268964767456054], 'transfer_latency': 0.004794547, 'config': {'audio_sr': 16000, 'freq_mask': 3000, 'model': 'hubert-large'}, 'placement': [1, 1, 1, 2], 'gpu_mem_usgs': [0, 0, 2000, 0], 'tot_gpu_time': 0.01975371837615967, 'type': 'random'}
{'utility': 7.138993910332716, 'gpu_time': 1.073906421661377, 'accuracy': 0.5181785062266171, 'latency': 0.02848926568322754, 'compute_latency_by_op': [0.3180861473083496, 0.01252896785736084, 0.01073906421661377, 0.04091677665710449], 'transfer_latency': 0.00701113725, 'config': {'audio_sr': 16000, 'freq_mask': 2000, 'model': 'wav2vec2-base'}, 'placement': [0, 2, 2, 2], 'gpu_mem_usgs': [0, 0, 1000, 0], 'tot_gpu_time': 0.01073906421661377, 'type': 'random'}
[[NodeState(remain_gpu_memory=1000, num_gpu_ps=0)], [NodeState(remain_gpu_memory=0, num_gpu_ps=1)], [NodeState(remain_gpu_memory=0, num_gpu_ps=2)]]
0
18.721686924589036
"""