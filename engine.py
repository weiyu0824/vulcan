from bo import BayesianOptimization
import docker 
import socket
import json 
import copy 
from task import speech_recognition_ops, speech_recongnition_knobs, Knob
from typing import Tuple, Dict, List
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



machine_infos = {
    "compute": [0.01, 0.22, 0.6, 1],
    "bandwidth": [0.08, 0.05, 0.01]
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

class Engine:
    def __init__(self, min_accuracy, max_latency, num_profile_sample=-1):
        self.early_prune_count = 5 
        self.bo_stop_count = 3
        self.num_init_sample_configs = 5
        self.utility_thold = 0  

        self.min_accuracy = min_accuracy
        self.max_latency = max_latency
        self.num_profile_sample = num_profile_sample

        # self.pipeline = Pipeline("video_analytics")
        # self.machines = [{"address": 'localhost'}]

        # self.docker_client = docker.from_env()

        self.port = 12343
        self.host = "localhost"

        self.knob_list = speech_recongnition_knobs
        self.ops = speech_recognition_ops

        with open("speech_recognition/profile_result_2.json", 'r') as fp:
            self.profile_result = json.load(fp)
        
        self.cache = {}

        self.feat_bounds = self.get_feat_bounds()

        self.profile_lat = 0        

    def start_tcp_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Bind the socket to the address and port
            server_socket.bind((self.host, self.port))

            # Listen for incoming connections
            server_socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            # Accept a new connection
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            # Recived container setup command
            setup_cmd = client_socket.recv(1024)
            print(setup_cmd)

            # Send start profiling command to the container
            client_socket.sendall(b"start profiling")
            
            # Received profiling stats
            stats = client_socket.recv(1024)
            print(stats)
            
            # Close the client socket
            client_socket.close() 
        
    def run_pipeline(self, config: dict):
        """
        Run the pipeline to get 1. the output_size of each operator & 2. overall accuracy
        """
        print("config: ", config)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Bind the socket to the address and port
            server_socket.bind((self.host, self.port))

            # Listen for incoming connections
            server_socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            # Open the container
            container = self.docker_client.containers.run(
                image="video-task",
                detach=True,  # Detach container so it runs in the background
                environment=config,  # Pass config as environment variables
                network_mode="host",  # Use host network for TCP connection
                volumes={
                    "/dev/shm/wylin/nuimages": {'bind': '/nuimages', 'mode': 'rw'},
                    "/users/wylin2/vulcan/video_analytics": {'bind': '/app', 'mode': 'rw'}
                },
                auto_remove=True,
                runtime="nvidia",  # Specify NVIDIA runtime for GPU support                
            )

            print(container.id)

            # Accept a new connection from the container
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            # Send start profiling command to the container
            client_socket.sendall(b"start profiling")
            
            # Received profiling stats
            stats = client_socket.recv(1024)
            stats = json.loads(stats)

            # Close the client socket
            client_socket.close()  
        


        return stats

    def simulate_run_pipeline(self, config: dict) -> dict:

        config_str = json.dumps(config)
        # print(config_str)
        if config_str in self.cache:
            return self.cache[config_str]
        
        for pf in self.profile_result:
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

    def postprocess_result(self, profile_result: dict, placement: list):
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

        assert len(self.ops) == len(placement)

        prev_op_machine = 0
        tot_compute_latency = 0
        tot_transfer_latency = 0
        tot_gpu_time = 0

        for op, op_machine in zip(self.ops, placement):
            op_name = op.name
            op_type = op.tpy
            compute_latency = profile_result[f"{op_name}_compute_latency"] / machine_infos['compute'][op_machine]

            transfer_latency = 0
            for machine_id in range(prev_op_machine, op_machine):
                bandwidth = machine_infos["bandwidth"][machine_id] #gbs
                bandwidth *= 10e9 #bytes/s
                transfer_latency += profile_result[f"{op_name}_input_size"] / bandwidth
            
            tot_compute_latency += compute_latency
            tot_transfer_latency += transfer_latency
            if op_type == 'dnn':
                tot_gpu_time += profile_result[f"{op_name}_compute_latency"] 

        pipe_accuracy = profile_result["cummulative_accuracy"][self.num_profile_sample - 1]
        pipe_latency = tot_compute_latency + tot_transfer_latency

        utility = (0.5 * -1 * (pipe_accuracy - self.min_accuracy) + 0.5 * (self.max_latency - pipe_latency)) / tot_gpu_time

        # print(pipe_latency, tot_compute_latency, tot_transfer_latency)
        # exit()
        post_result = {
            "utility": utility,
            "gpu_time": tot_gpu_time * 100,
            "accuracy": pipe_accuracy,
            "latency": pipe_latency  
        }
       
        return post_result
    
    def generate_all_placement_choices(self):
        num_ops = len(self.ops)
        num_tier = len(machine_infos["compute"])

        all_choices = []

        def dfs(cur_placements):
            if len(cur_placements) >= num_ops:
                all_choices.append(cur_placements)
                return

            prev_placement = 0
            if len(cur_placements) > 0:
                prev_placement = cur_placements[-1]
            for p in range(prev_placement, num_tier):
                dfs(cur_placements + [p])

        dfs([])
        return all_choices 

    def generate_all_configs(self):
        num_knobs = len(self.knob_list)

        all_choices = []

        def dfs(cur_config):
            cur_num_knob = len(cur_config)
            if cur_num_knob >= num_knobs:
                all_choices.append(cur_config)
                return

            knob_name = self.knob_list[cur_num_knob].name
            for knob_choice in self.knob_list[cur_num_knob].choice:
                cur_config[knob_name] = knob_choice
                dfs(copy.deepcopy(cur_config))
                cur_config.pop(knob_name)

        dfs({})
        return all_choices
    
    def get_feat_bounds(self):
        feat_bounds = []
        for knob in self.knob_list:
            feat_bounds.append([i for i in range(len(knob.choice))]) 
        
        return feat_bounds

    # 
    def my_cbo(self):
        self.cache = {}
        placements: list[list] = self.generate_all_placement_choices()
        foot_print = []

        
        # all_samples, all_accs = [], []

        for placement in placements:
            acc_bo = BayesianOptimization(feat_bounds=self.feat_bounds)
            lat_bo = BayesianOptimization(feat_bounds=self.feat_bounds)
            util_bo = BayesianOptimization(feat_bounds=self.feat_bounds)

            # init BO 
            # 1. Random sample configs
            init_samples = acc_bo.get_random_samples(num_samples=self.num_init_sample_configs)
            init_utils, init_accs, init_lats =  [], [], []
            # 2. Run those config to get ACC & Latency
            for sample in init_samples:
                
                # Convert BO sample back to config
                config = sample_to_config(sample, self.knob_list)

                result = self.simulate_run_pipeline(config=config)

                util, lat = self.postprocess_result(result, placement)
                
                init_lats.append(lat * -1)
                init_accs.append(result["accuracy"] * -1) 
                init_utils.append(util)
            


                foot_print.append((util, lat, result["accuracy"], config, placement, 'random'))
            
            # all_samples += init_samples
            acc_bo.fit(X=init_samples, y=init_accs)
            lat_bo.fit(X=init_samples, y=init_lats)
            util_bo.fit(X=init_samples, y=init_utils)

            # print('random', all_samples, all_accs, init_lats)
            
            # BO optimization 
            prune_count = 0  
            prev_utility = max(init_utils)
            stop_count = 0
            while True:
                # select highest utils with guarantee acc & lat
                sorted_sample = util_bo.get_sorted_samples()

                sample = sorted_sample[0] 
        
                for ss in sorted_sample:
                    
                    acc_ub = acc_bo._get_acq_val(ss) * -1
                    lat_ub = lat_bo._get_acq_val(ss) * -1
                    # print('ask!, pred: ', acc_ub, lat_ub)
                    if acc_ub < self.min_accuracy and lat_ub < self.max_latency:
                        sample = ss
                        break
                # exit(0)
            

                config = sample_to_config(sample, self.knob_list) 

                config_str = json.dumps(config)
                if config_str in self.cache:
                    result = self.cache[config_str]
                else:
                    result = self.simulate_run_pipeline(config=config)
                    self.cache[config_str] = result

                util, lat = self.postprocess_result(result, placement) 
                # print(f"Utility: {utility}, Latency: {appox_latency}, Accuracy: {result["accuracy"]}, Config: {config}")

                # print('pred (u, a, l): ', util_bo._get_acq_val(sample), acc_bo._get_acq_val(sample)* -1, lat_bo._get_acq_val(sample)*-1)
                # print('true (u, a, l): ', util, result["accuracy"], lat )
                # print('----------')

                # Fit BO again
                init_samples.append(sample)
                init_accs.append(result["accuracy"] * -1)
                # print('pred: ', acc_bo._get_acq_val(sample), 'true: ', result["accuracy"])
                init_lats.append(lat * -1)
                init_utils.append(util)

                # all_samples.append(sample)
                acc_bo.fit(X=init_samples, y=init_accs)
                lat_bo.fit(X=init_samples, y=init_lats)
                util_bo.fit(X=init_samples, y=init_utils)

                # threshold = min utility to meet requirement = 0,
                # since (Am-Am) + (Lm-Lm) = 0 
                # 1. Early stop
                if util <= self.utility_thold:
                    prune_count += 1
                    if prune_count >= self.early_prune_count:
                        # print('early prune')
                        break
                
                foot_print.append((util, lat, result["accuracy"], config, placement))

                # 2. BO stop
                if util < prev_utility * 1.1:
                    stop_count += 1
                else:
                    stop_count = 0 
                if stop_count >= self.bo_stop_count:
                    # print('bo stop')
                    break
                prev_utility = max(prev_utility, util)

            # print('cache', len(self.cache))
            # utilities.sort(key=lambda x: x[0])
            # print('best', utilities[len(utilities) - 1])
            # exit()
        foot_print.sort(key=lambda x: x[0], reverse=True)
        f_foot_print = []
        for d in foot_print:
            if d[1] < self.max_latency and d[2] < self.min_accuracy:
                f_foot_print.append(d)

        f_foot_print.sort(key=lambda x: x[0], reverse=True)
        if len(f_foot_print) <= 0:
            return foot_print[0], 0
        return f_foot_print[0], 0 
              
    def vulcan_search(self, optimize=False):
        self.cache = {}
        self.profile_lat = 0

        placements: list[list] = self.generate_all_placement_choices()
        foot_print = []
        # total_profile_time = 0
        for placement in placements:
            # if placement != [2, 2, 3, 3]:
            #     continue
            # bo
            bo = BayesianOptimization(feat_bounds=self.feat_bounds) 
            init_utils = [] # utility

            # opt: 
            init_samples, init_utils =  [], []
            
            if optimize and len(self.cache) >= 3 :
                for config_str, result in self.cache.items():
                    config = json.loads(config_str)
                    post_result = self.postprocess_result(result, placement)
                    sample = config_to_sample(config, self.knob_list) 
                    #
                    init_samples.append(sample)
                    init_utils.append(post_result['utility'])
                    # utilities.append((utility, appox_latency, result["accuracy"], config, placement, 'prev'))

                bo.fit(X=init_samples, y=init_utils)
            else: 
                
                # init BO 
                # 1. Random sample configs
                init_samples = bo._get_random_raw_samples(num_samples=self.num_init_sample_configs)
                init_utils = []
                # 2. Run those config to get ACC & Latency
                for sample in init_samples:
                    
                    config = sample_to_config(sample, self.knob_list)
                    result = self.simulate_run_pipeline(config=config)  

                    post_result = self.postprocess_result(result, placement)
                    init_utils.append(post_result['utility'])

                    foot_print.append({
                        'gpu_time': post_result['gpu_time'],
                        'latency': post_result['latency'],
                        'accuracy': post_result['accuracy'],
                        'config': config,
                        'placement': placement,
                        'utility': post_result['utility'],
                        'type': 'random'
                    })
                    
                bo.fit(X=init_samples, y=init_utils)


            prune_count, stop_count = 0, 0  
            max_util = max(init_utils)
            # continue
            while True:
                sample = bo.get_next_sample()
                config = sample_to_config(sample, self.knob_list) 

                result = self.simulate_run_pipeline(config=config)

                post_result = self.postprocess_result(result, placement) 


                # Fit BO again
                init_samples.append(sample)
                init_utils.append(post_result['utility'])
                bo.fit(X=init_samples, y=init_utils)
                foot_print.append({
                    'gpu_time': post_result['gpu_time'],
                    'latency': post_result['latency'],
                    'accuracy': post_result['accuracy'],
                    'config': config,
                    'placement': placement,
                    'utility': post_result['utility'],
                    'type': 'search'
                })

                # print(config)

                prune_count += 1 if post_result['utility'] < self.utility_thold else 0
                stop_count += 1 if post_result["utility"] < max_util else 0

                max_util = max(post_result["utility"], max_util)

                if prune_count >= self.early_prune_count or stop_count > self.bo_stop_count:
                    break

            # print('cache', len(self.cache))
            # utilities.sort(key=lambda x: x[0])
            # print('best', utilities[len(utilities) - 1])
            # exit()
        # print(foot_print)
        foot_print.sort(key=lambda x: x['utility'], reverse=True) 
        # filter:
        f_foot_print = []
        for d in foot_print:
            if d['latency'] < self.max_latency and d['accuracy'] < self.min_accuracy:
                f_foot_print.append(d)

        if len(f_foot_print) <= 0:
            print('none')
            return foot_print[0], self.profile_lat
        
        return f_foot_print[0], self.profile_lat

    def exhaustive_search_best_placement(self):
        self.cache = {}
        self.profile_lat = 0

        placements: list[list] = self.generate_all_placement_choices()
        configs: List[dict] = self.generate_all_configs()
        foot_print = [] 
        for placement in placements: 
            # if placement != [2, 2, 3, 3]:
            #     continue
            for config in configs:
                result = self.simulate_run_pipeline(config)
                post_result = self.postprocess_result(result, placement)
                foot_print.append({
                    'gpu_time': post_result['gpu_time'],
                    'latency': post_result['latency'],
                    'accuracy': post_result['accuracy'],
                    'config': config,
                    'placement': placement,
                    'type': 'search'
                })

        foot_print.sort(key=lambda x: x['gpu_time']) 
        # filter:
        f_foot_print = []
        for d in foot_print:
            if d['latency'] < self.max_latency and d['accuracy'] < self.min_accuracy:
                f_foot_print.append(d)

        if len(f_foot_print) <= 0:
            return foot_print[0], self.profile_lat
        # if len(f_foot_print) > 3:
        #     print(f_foot_print[0])
        #     print(f_foot_print[1])
        #     print(f_foot_print[2])
        return f_foot_print[0], self.profile_lat
    
    def random_iterative(self, n_sample=50):
        self.cache = {}
        bo = BayesianOptimization(feat_bounds=self.feat_bounds)
        samples = bo.get_random_samples(n_sample)
        placements = self.generate_all_placement_choices()

        foot_print = []
        for placement in placements: 
            for sample in samples:
                config = sample_to_config(sample, self.knob_list) 
                result = self.simulate_run_pipeline(config)
                util, lat = self.postprocess_result(result, placement)
                foot_print.append((util, lat, result["accuracy"], config, placement))
        

        foot_print.sort(key=lambda x: x[0], reverse=True)

        f_foot_print = []
        for d in foot_print:
            if d[1] < self.max_latency and d[2] < self.min_accuracy:
                f_foot_print.append(d)
        
        f_foot_print.sort(key=lambda x: x[0], reverse=True)

        if len(f_foot_print) == 0:
            return foot_print[0], 0
        else: 
            return f_foot_print[0], 0

    def all_bo(self):
        self.cache = {}
        possible_placements = self.generate_all_placement_choices()
        feat_bounds = copy.deepcopy(self.feat_bounds)
        n_tiers = 4
        for _ in range(len(self.ops)):
            feat_bounds.append([i for i in range(n_tiers)])
        
        bo = BayesianOptimization(feat_bounds=feat_bounds)
        # print(feat_bounds)

        sorted_sample = bo.get_sorted_samples()

        all_samples = []
        utils = []
        foot_print = []
        for sample in sorted_sample:
            config = sample_to_config(sample[:len(self.knob_list)], self.knob_list)
            placement = sample[len(self.knob_list):]
            if placement in possible_placements:
                all_samples.append(sample)
                result = self.simulate_run_pipeline(config=config)

                util, lat = self.postprocess_result(result, placement) 
                # print(f"Utility: {util}, Latency: {lat}, Accuracy: {result["accuracy"]}, Config: {config}")
                utils.append(util) 
                foot_print.append((util, lat, result["accuracy"], config, placement))

            if len(all_samples) > 10:
                break

        bo.fit(X=all_samples, y=utils)
        for i in range(50):
            
            sorted_sample = bo.get_sorted_samples()
            for sample in sorted_sample:
                config = sample_to_config(sample[:len(self.knob_list)], self.knob_list)
                placement = sample[len(self.knob_list):]
                if placement in possible_placements:
                    all_samples.append(sample)
                    result = self.simulate_run_pipeline(config=config)

                    util, lat = self.postprocess_result(result, placement) 
                    # print(f"Utility: {util}, Latency: {lat}, Accuracy: {result["accuracy"]}, Config: {config}")
                    utils.append(util) 
                    foot_print.append((util, lat, result["accuracy"], config, placement))



                    break 
            bo.fit(X=all_samples, y=utils)        

        foot_print.sort(key=lambda x: x[0], reverse=True)

        f_foot_print = []
        for d in foot_print:
            if d[1] < self.max_latency and d[2] < self.min_accuracy:
                f_foot_print.append(d)

        f_foot_print.sort(key=lambda x: x[0], reverse=True)
        if len(f_foot_print) <= 0:
            return foot_print[0], 0
        return f_foot_print[0], 0  
    
    def constraint_bo(self):
        self.cache = {}
        self.profile_lat = 0

        placements: list[list] = self.generate_all_placement_choices()
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
    
    def bo(self):
        pass

if __name__ == "__main__":

    engine = Engine(min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    for i in range(10):
        result, profile_lat = engine.constraint_bo()
        print(result)
        print(len(engine.cache))
        exit()
    # # print('---------------')
    # # result, profile_lat = engine.exhaustive_search_best_placement()
    # # print(result)
    # for i in range(10):
    #     result, profile_lat = engine.vulcan_search()
    #     print(result)
    #     print(len(engine.cache))

    # exit(0)
   
    exp_results = []
    def run(method, num_trails, min_accuracy, max_latency, num_profile_sample):
        engine = Engine(min_accuracy=min_accuracy, max_latency=max_latency, num_profile_sample=num_profile_sample)

        oracle_engine = Engine(min_accuracy=min_accuracy, max_latency=max_latency, num_profile_sample=5000)
        num_qualified_trail = 0
        cache_size = []
        best_utils = []
        oracle_utils = []
        profile_overheads = []
        for _ in range(num_trails):
            if method == 'vulcan': 
                result, profile_lat = engine.vulcan_search()
            elif method == 'vulcan+acc_prior':
                result, profile_lat = engine.vulcan_search(optimize=True)
            elif method == 'random': 
                result, profile_lat = engine.random_iterative(50)
            elif method == 'exhaustive':
                result, profile_lat = engine.exhaustive_search_best_placement()
            elif method == 'cbo':
                result, profile_lat = engine.constraint_bo()
            else: 
                raise KeyError('Unsupported key')
                return

            cache_size.append(len(engine.cache))
            best_utils.append(result['utility'])

            profile_overheads.append(profile_lat)
            print(profile_lat)
            if result['latency'] <=  max_latency and result['accuracy'] <= min_accuracy: 
                num_qualified_trail += 1
            
            # print(placement)
            c = result['config']
            p = result['placement']
            # print(c, p)
            profile_stats = oracle_engine.simulate_run_pipeline(c)
            oracle_util = oracle_engine.postprocess_result(profile_stats, p)['utility']
            oracle_utils.append(oracle_util)
            # exit()

        exp_results.append( dict(
            optimization_type=method,
            min_accurac=min_accuracy,
            max_latency=max_latency, 
            num_profile_sample=num_profile_sample,
            num_trails=num_trails,
            qualify_optimization=num_qualified_trail,
            cache_size=cache_size,
            best_utils=best_utils, 
            profile_overheads=profile_overheads,
            oracle_utils=oracle_utils
        ))
        print(f'{method} \
                #qualify optimization: {num_qualified_trail}/{num_trails} \
                avg cache size: {sum(cache_size)/num_trails} \
                avg utility: {sum(best_utils)/num_trails}' )
    
        
    # 100, 300, 500, 1000, 2000, 3000, 4000, 5000,  
    run('vulcan', num_trails=20, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    run('cbo', num_trails=20, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)

    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=300)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=500)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=1000)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=2000)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=3000)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=4000)
    # run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    # run('exhaustive',  num_trails=1, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    # # run('acc_lat_bo_const')

    # with open('opt_result.json', 'w') as fp:
    #     json.dump(result, fp)