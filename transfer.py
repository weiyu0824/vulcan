from task import speech_recognition_ops, speech_recongnition_knobs, Knob, Operator, speech_recognition_dnn_usg
from data import Query, QuerySetup, TierSpec, SetupMetric
from utils import get_feat_bounds, probability_greater_than_t, sample_to_config, config_to_sample
import json
import copy 
import numpy as np
from bo import BayesianOptimization
from typing import List
from cluster import ClusterState
np.random.seed(5)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

mach_info = {
    "compute": [0.1, 0.22, 1],
    "bandwidth": [0.08, 0.05, 0], # bw=(t0->t1, t1->t2, t2->t3)
    "cost": [1, 2, 3],
}

def transform_profile_result(query: Query, profile_results) -> dict:
    transformed = {}
    for pf in profile_results:
        config = {}
        for knob in query.knob_list:
            config[knob.name] = pf[knob.name]
        results = pf['result']
        # print(results)
        for key, value in results.items():
            if type(value) == float:
                results[key] = round(value, 2)
        transformed[json.dumps(config)] = pf['result']

    return transformed

maximum_step = 10
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
                    profile_results = json.load(fp)  
                    profile_results = transform_profile_result(query, profile_results)      
                    # print(profile_results)
                    # exit()
                self.profile_result_cache[query.profile_result_file] = profile_results   
        
        self.previous_acc_bos = []
        self.previous_lat_bos = []
        self.step_limit = 40
             
    def simulate_pipeline(self, config: dict, query: Query) -> dict:

        # config_str = json.dumps(config)
        # if config_str in self.cache:
        #     return self.cache[config_str]

        # profile_result = self.profile_result_cache[query.profile_result_file]
        # for pf in profile_result:
        #     found = True
        #     for knob_name, knob_val in config.items():
        #         if pf[knob_name] != knob_val:
        #             found = False
        #             break
        #     if found:
        #         self.cache[config_str] = pf["result"]
        #         self.profile_lat += pf["result"]["total_profile_time"] * (self.num_profile_sample / len(pf["result"]["cummulative_accuracy"]))
        #         return pf["result"]

        # return {}
        config_str = json.dumps(config)
        profile_result = self.profile_result_cache[query.profile_result_file][config_str]
        self.cache[config_str] = profile_result 
        return self.cache[config_str]

    def online_profile_pipeline(self, config: dict, query: Query):

        pass

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

        # pipe_accuracy = profile_result["cummulative_accuracy"][self.num_profile_sample - 1]
        pipe_accuracy = profile_result['accuracy']
        pipe_latency = tot_compute_latency + tot_transfer_latency

        #TODO: This is not general
        #cause we know that current pipeline only use GPU in op3
        cost = mach_info['cost'][placement[3]] * tot_gpu_time 

        utility = (0.5 * -1 * (pipe_accuracy - query.acc_thold) + 0.5 * (query.lat_thold - pipe_latency)) / cost
        acc_index = -1 * (pipe_accuracy - query.acc_thold) / cost
        # print(placement, mach_info['compute'])
        

        setup_metric = SetupMetric(
            accuracy=round(pipe_accuracy, 2),
            latency=round(pipe_latency, 2),
            utility=round(utility, 2),
            cost = round(cost, 2),
            tot_transfer_latency=round(tot_transfer_latency, 2),
            tot_gpu_time=tot_gpu_time,
            accuracy_index=acc_index,
            compute_latency_by_op=[], #TODO:do we need this metric?
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
                    # print(f"{prune_count}/{self.early_prune_count}, {stop_count}/{self.bo_stop_count}")
                    break
                if len(self.cache) >= self.step_limit:
                    return query_setups
        return query_setups  

    def search_by_transfer(self, query: Query, cluster_state: ClusterState):
        """
        """

        self.cache = {}
        self.profile_lat = 0
        feat_bounds = get_feat_bounds(query) 
        placements = self.generate_placement_choices(query, cluster_state.spec)

        query_setups: List[QuerySetup] = []
        # print(feat_bounds)
        # TODO: Rewrite this part of code.
        # hard to read
        # place_bounds = get_feat_bounds(query)
        # for _ in range(len(placements[0])):
        #     place_bounds.append([0, 1, 2])
        
        # total_profile_time = 0
        acc_bo = BayesianOptimization(feat_bounds=feat_bounds)
        lat_bo = BayesianOptimization(feat_bounds=feat_bounds)
        self.cache = {}
        def get_next_sample(num_samples, is_init_point=True):
            if len(self.previous_acc_bos) == 0 and is_init_point:
                return acc_bo._get_random_raw_samples(num_samples=3)
            
            if len(self.previous_acc_bos) != 0 and is_init_point:
                ref_acc_bo = self.previous_acc_bos[0]
                ref_lat_bo = self.previous_lat_bos[0]
                # print('r1', ref_acc_bo.feat_bounds, ref_lat_bo.feat_bounds)
            else:
                ref_acc_bo = acc_bo
                ref_lat_bo = lat_bo
                # print('r2', ref_acc_bo.feat_bounds, ref_lat_bo.feat_bounds)

            samples: List[List[int]] = ref_acc_bo._get_random_raw_samples(num_samples=1000)
            weighted_acq_vals = []
            sample_vals = []
            placement_vals = []
            visited_sample = set()
            for c_sample in samples:
                if tuple(c_sample) in visited_sample:
                    continue
                visited_sample.add(tuple(c_sample))
                if json.dumps(sample_to_config(c_sample, query.knob_list)) in self.cache:
                    continue
                pred_acc_mean, pred_acc_std = ref_acc_bo.get_pred(c_sample)
                prob_meet_acc = 1 - probability_greater_than_t(pred_acc_mean, pred_acc_std, query.acc_thold)


                for placement in placements:
                    pred_lat_mean, pred_lat_std = ref_lat_bo.get_pred(c_sample + placement)
                    prob_meet_lat = 1 - probability_greater_than_t(pred_lat_mean, pred_lat_std, query.lat_thold)
                
                    # tmp 
                    # predict cost given this config
                    # save = 10000 - cost
                    # TODO: this code is not general
                    # price = sum(placement * mach_info["compute"])
                    save = 10000 - speech_recognition_dnn_usg[sample_to_config(c_sample, query.knob_list)['model']] * mach_info['compute'][placement[3]] 

                    # print('----')
                    # print(pred_acc_mean, pred_acc_std, self.min_accuracy)
                    # print(pred_lat_mean, pred_lat_mean, self.max_latency)
                    # print(prob_meet_acc, prob_meet_lat)

                    w_acq_val = save * prob_meet_acc * prob_meet_lat 
                    weighted_acq_vals.append(w_acq_val)
                    sample_vals.append(c_sample)
                    placement_vals.append(placement)

            poss_samples = [s for _, s, p in sorted(zip(weighted_acq_vals, sample_vals, placement_vals), reverse=True)]
            next_samples = [] 
            i = 0
            while len(next_samples) < num_samples:
                if (poss_samples[i] not in next_samples):
                    next_samples.append(poss_samples[i])
                i += 1
            return next_samples

        # opt: using history by other placements
        init_samples: List[List[int]] = []
        init_utils: List[float] = []

        # BO Intialization 
        # 1. Random sample configs
        # init_samples = get_next_sample(self.num_init_sample_configs)
        init_samples = get_next_sample(num_samples=3, is_init_point=True)
        init_accs = []
        init_lats = []
        init_lat_samples = []
        # 2. Run those config to get ACC & Latency
        for sample in init_samples:
            print(sample)
            
            config = sample_to_config(sample, query.knob_list)
            profile_result = self.simulate_pipeline(config=config, query=query)
            add_acc = False
            for placement in placements:
                setup_metric = self.postprocess_result(profile_result, placement, config, query)
                init_lat_samples.append(sample + placement)
                init_lats.append(setup_metric.latency)
                query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))
                if not add_acc:
                    init_accs.append(setup_metric.accuracy)
                    add_acc = True

        acc_bo.fit(X=np.array(init_samples), y=np.array(init_accs))
        lat_bo.fit(X=np.array(init_lat_samples), y=np.array(init_lats))

        # BO Searching
        prune_count, stop_count = 0, 0  
        # max_util = max(init_utils)

        exp_result = []
        while len(self.cache) < self.step_limit:
            sample = get_next_sample(num_samples=1)[0]
            init_samples.append(sample)
            config = sample_to_config(sample, query.knob_list) 
            profile_result = self.simulate_pipeline(config=config, query=query)

            add_acc = False
            for placement in placements:
                setup_metric = self.postprocess_result(profile_result, placement, config, query)
                init_lat_samples.append(sample + placement)
                init_lats.append(setup_metric.latency)
                query_setups.append(QuerySetup(query=query, placement=placement, config=config, setup_metric=setup_metric))
                if not add_acc:
                    init_accs.append(setup_metric.accuracy)
                    add_acc = True

            # Fit BO again
            acc_bo.fit(X=np.array(init_samples), y=np.array(init_accs))
            lat_bo.fit(X=np.array(init_lat_samples), y=np.array(init_lats))


            # prune_count += 1 if setup_metric.utility < self.utility_thold else 0
            # stop_count += 1 if setup_metric.utility < max_util else 0

            # max_util = max(setup_metric.utility, max_util)

            # if prune_count >= self.early_prune_count or stop_count > self.bo_stop_count:
            #     break

            ex_query_setups = filter(lambda x: x.setup_metric.accuracy <= query.acc_thold and x.setup_metric.latency <= self.queries[0].lat_thold  ,query_setups)
            ex_query_setups = sorted(ex_query_setups, key=lambda x: x.setup_metric.cost, reverse=False)
            if len(ex_query_setups) > 1:
                # print(f"Step: {len(self.cache)}")
                # print(ex_query_setups[0].config)
                # print(ex_query_setups[0].placement)
                # print(ex_query_setups[0].setup_metric)
                exp_result.append(ex_query_setups[0].setup_metric.cost) 
            else:
                # print('Not found')
                exp_result.append(0) 

        self.previous_acc_bos.append(acc_bo)
        self.previous_lat_bos.append(lat_bo)
        print()
        print('Exp result:', exp_result)

        return query_setups
      
    def run(self):
        for qid in range(len(queries)):
            query_setups = self.search_by_transfer(self.queries[2], ClusterState(self.cluster_spec))
            print('------')
            
            query_setups = filter(lambda x: x.setup_metric.accuracy <= self.queries[qid].acc_thold and x.setup_metric.latency <= self.queries[0].lat_thold  ,query_setups)
            query_setups = sorted(query_setups, key=lambda x: x.setup_metric.cost, reverse=False)
            # if len(query_setups) > 1:
            #     # print(query_setups[0].config)
            #     # print(query_setups[0].placement)
            #     # print(query_setups[0].setup_metric) 
            # else:
            #     print('Not found')
            print('cache size', len(self.cache))
            # exit()

if __name__ == "__main__":
    queries = [
        Query(0.3, 0.3, "speech_recognition/profile_result_libri.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        Query(0.4, 0.3, "speech_recognition/profile_result_voices.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg),
        Query(0.2, 0.3, "speech_recognition/profile_result_lium.json", speech_recongnition_knobs, speech_recognition_ops, speech_recognition_dnn_usg), 
    ]
    cluster_spec = [
        TierSpec('edge', num_nodes=1, num_gpus=1, gpu_memory_size=1000, compute_speed=mach_info['compute'][0], bandwidth=mach_info['bandwidth'][0]),
        TierSpec('on-perm', num_nodes=1, num_gpus=1, gpu_memory_size=2000, compute_speed=mach_info['compute'][1], bandwidth=mach_info['bandwidth'][1]),
        TierSpec('cloud', num_nodes=1, num_gpus=1, gpu_memory_size=3000, compute_speed=mach_info['compute'][2], bandwidth=mach_info['bandwidth'][2]),
    ]

    engine = Engine(queries, cluster_spec)
    engine.run()
    # path_samples, path_accs = search_pereto_climb(queries[0])
    # acc_bo.fit(path_samples, path_accs)
    # print('[q1]')
    # print('init from history:') 
    # path_samples, path_accs = search_pereto_climb(queries[1], acc_bo)
    # print('random init:')
    # path_samples, path_accs = search_pereto_climb(queries[1])
    # print('[q2]')
    # print('init from history:')
    # path_samples, path_accs = search_pereto_climb(queries[2], acc_bo)
    # print('random init:')
    # path_samples, path_accs = search_pereto_climb(queries[2])
    # search_pereto_exhaustive(queries[0])