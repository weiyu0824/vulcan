from bo import BayesianOptimization
import docker 
import socket
import json 
import copy 

# perf_weight = 0.5
# # hyper-params:
# accuracy_coef = 1
# latency_coef = 1
# compute_coef = 1
# network_coef = 1
# g:
# aA:
# aL:
# agpu:
# anet:
from typing import Tuple, Dict, List

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# video_analytics_knobs = [
#     ("models", ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']),
#     ("resize_factor", [0.9, 0.8, 0.7, 0.6, 0.5])
# ]
speech_recongnition_knobs = [
    ("audio_sr", [8000, 10000, 12000, 14000, 16000]),
    ("freq_mask", [500, 1000, 2000, 3000, 4000]),
    ("model", ["wav2vec2-base", "wav2vec2-large-10m", "wav2vec2-large-960h", "hubert-large", "hubert-xlarge"])
]
speech_recognition_ops = ["audio_sampler", "noise_reduction", "wave_to_text", "decoder"]

machine_infos = {
    "compute": [0.01, 0.22, 0.6, 1],
    "bandwidth": [0.08, 0.05, 0.01]
}

def sample_to_config(sample: List[int], knob_list)-> dict: 
    config = {}
    assert len(sample) == len(knob_list), "Sample and should have equal size as knob"

    for knob_idx, (knob_name, knob_choices) in enumerate(knob_list): 
        config[knob_name] = knob_choices[sample[knob_idx]]
    return config
def config_to_sample(config: dict, knob_list) -> List[int]:
    sample = []
    for knob_name, knob_choices in knob_list:  
        i = knob_choices.index(config[knob_name])
        sample.append(i)
    return sample

class Engine:
    def __init__(self, min_accuracy, max_latency, num_profile_sample=-1):
        self.early_prune_count = 3
        self.bo_stop_count = 5
        self.num_init_sample_configs = 3
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

    def calculate_utility(self, profile_stats: dict, placement: list) -> Tuple[float, float]:
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
        for op, op_machine in zip(self.ops, placement):
            compute_latency = profile_stats[f"{op}_compute_latency"] / machine_infos['compute'][op_machine]

            transfer_latency = 0
            for machine_id in range(prev_op_machine, op_machine):
                bandwidth = machine_infos["bandwidth"][machine_id] #gbs
                bandwidth *= 10e9 #bytes/s
                transfer_latency += profile_stats[f"{op}_input_size"] / bandwidth
            
            tot_compute_latency += compute_latency
            tot_transfer_latency += transfer_latency

        # pipe_accuracy = profile_stats["accuracy"]_
        pipe_accuracy = profile_stats["cummulative_accuracy"][self.num_profile_sample - 1]
        pipe_latency = tot_compute_latency + tot_transfer_latency

        utility = 0.5 * -1 * (pipe_accuracy - self.min_accuracy) + 0.5 * (self.max_latency - pipe_latency) 

        # if pipe_accuracy > self.min_accuracy or pipe_latency > self.max_latency:
        #     utility = 0
       
        return utility, pipe_latency
    
        # mbs_per_sec = 20
        # cpu_gpu_processing_diff = 25
        # appox_latency = 0
        # appox_latency += profile_stats["loader_compute_latency"] 
        # if placement == [0, 1]: # edge -> cloud
        #     appox_latency += profile_stats["loader_output_size"] / (1e6 * mbs_per_sec)
        #     appox_latency += profile_stats["detector_compute_latency"]
        # else: # edge -> edge
        #     appox_latency += profile_stats["detector_compute_latency"] * cpu_gpu_processing_diff 

        # utility = 0.5 * (profile_stats["accuracy"] - self.min_accuracy) + 0.5 * (self.max_latency - appox_latency)

        # return utility, appox_latency

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

            knob_name = self.knob_list[cur_num_knob][0]
            for knob_choice in self.knob_list[cur_num_knob][1]:
                cur_config[knob_name] = knob_choice
                dfs(copy.deepcopy(cur_config))
                cur_config.pop(knob_name)

        dfs({})
        return all_choices
    
    def get_feat_bounds(self):

        feat_bounds = []
        for _, knob_choice in self.knob_list:
            feat_bounds.append([i for i in range(len(knob_choice))]) 
        
        
        return feat_bounds

    def experiment2(self):
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

                util, lat = self.calculate_utility(result, placement)
                
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

                util, lat = self.calculate_utility(result, placement) 
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
              
    def select_best_placement(self, optimize=False):
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
            init_utilities = [] # utility

            # opt: 
            init_samples, init_utilities =  [], []
            
            if optimize and len(self.cache) >= 3 :
                for config_str, result in self.cache.items():
                    config = json.loads(config_str)
                    utility, appox_latency = self.calculate_utility(result, placement)
                    sample = config_to_sample(config, self.knob_list) 
                    #
                    init_samples.append(sample)
                    init_utilities.append(utility)
                    # utilities.append((utility, appox_latency, result["accuracy"], config, placement, 'prev'))

                bo.fit(X=init_samples, y=init_utilities)
            else: 
                
                # init BO 
                # 1. Random sample configs
                init_samples = bo.get_random_samples(num_samples=self.num_init_sample_configs)
                init_utilities = []
                # 2. Run those config to get ACC & Latency
                for sample in init_samples:
                    
                    # Convert BO sample back to config
                    # print(sample)
                    config = sample_to_config(sample, self.knob_list)
                    result = self.simulate_run_pipeline(config=config)  
                    # total_profile_time += result["total_profile_time"] 

                    utility, appox_latency = self.calculate_utility(result, placement)
                    init_utilities.append(utility)

                    foot_print.append((utility, appox_latency, result["accuracy"], config, placement, 'random'))
                    # print((utility, appox_latency, result["accuracy"], config, placement, 'random'))

                # print(len(init_samples), len(init_utilities))
                # 3. Fit those samples into BO 
                bo.fit(X=init_samples, y=init_utilities)

            # BO optimization 
            prune_count = 0  
            prev_utility = max(init_utilities)
            attempt_idx = 0 
            stop_count = 0
            # continue
            while True:
                # print("Attempt: ", attempt_idx, 'stop cnt: ', stop_count, 'prune cnt: ', prune_count, 'prev_utility: ', prev_utility)
                attempt_idx += 1
                sample = bo.get_next_sample()
                config = sample_to_config(sample, self.knob_list) 

                result = self.simulate_run_pipeline(config=config)
                # total_profile_time += result["total_profile_time"]

                utility, appox_latency = self.calculate_utility(result, placement) 
                # print(f"Utility: {utility}, Latency: {appox_latency}, Accuracy: {result["accuracy"]}, Config: {config}")


                # Fit BO again
                init_samples.append(sample)
                init_utilities.append(utility)
                bo.fit(X=init_samples, y=init_utilities)

                # threshold = min utility to meet requirement = 0,
                # since (Am-Am) + (Lm-Lm) = 0 
                # 1. Early stop
                if utility <= self.utility_thold:
                    prune_count += 1
                    if prune_count >= self.early_prune_count:
                        # print('early prune')
                        break
                
                foot_print.append((utility, appox_latency, result["accuracy"], config, placement))

                # 2. BO stop
                if utility < prev_utility * 1.1:
                    stop_count += 1
                else:
                    stop_count = 0 
                if stop_count >= self.bo_stop_count:
                    # print('bo stop')
                    break
                prev_utility = max(prev_utility, utility)

            # print('cache', len(self.cache))
            # utilities.sort(key=lambda x: x[0])
            # print('best', utilities[len(utilities) - 1])
            # exit()

        foot_print.sort(key=lambda x: x[0], reverse=True) 
        # filter:
        f_foot_print = []
        for d in foot_print:
            if d[1] < self.max_latency and d[2] < self.min_accuracy:
                f_foot_print.append(d)

        f_foot_print.sort(key=lambda x: x[0], reverse=True)
        if len(f_foot_print) <= 0:
            return foot_print[0], self.profile_lat
        # print(foot_print[:3])
        return f_foot_print[0], self.profile_lat

    def experiment(self):
        # find accuracy 
        placement = [2, 2, 3, 3] 
        bo = self.init_bo()
        total_profile_time = 0
        init_samples, init_utilities, utilities = [], [], []
        prune_count = 0  
        prev_utility = -100
        attempt_idx = 0 
        stop_count = 0
        # continue

        for _ in range(10):
            sample = bo.get_next_sample()
            config = sample_to_config(sample, self.knob_list) 
            result = self.simulate_run_pipeline(config=config)

            utility, appox_latency = self.calculate_utility(result, placement) 

            print(f"Utility: {utility}, Latency: {appox_latency}, Accuracy: {result["accuracy"]}, Config: {config}")

            val = result["accuracy"] * -1
            # val = utility
            # val = appox_latency * -1

            # Fit BO again
            init_samples.append(sample)
            init_utilities.append(val)
            bo.fit(X=init_samples, y=init_utilities)
            utilities.append((utility, appox_latency, result["accuracy"], config, placement))

    def exhaustive_search_best_placement(self):
        self.cache = {}
        placements: list[list] = self.generate_all_placement_choices()
        configs: List[dict] = self.generate_all_configs()
        utilities = []
        for placement in placements: 
            # if placement != [2, 2, 3, 3]:
            #     continue
            for config in configs:
                result = self.simulate_run_pipeline(config)
                utility, appox_latency = self.calculate_utility(result, placement)

                if appox_latency < self.max_latency and result["accuracy"] < self.min_accuracy:
                    utilities.append((utility, appox_latency, result["accuracy"], config, placement))

        # def print_list(arr):
        #     for e in arr: 
        #         print(e)
        # utilities.sort(key=lambda x: x[0], reverse=True)
        # print('--- top utility ---')
        # print_list(utilities[:20])
        # utilities.sort(key=lambda x: x[1], reverse=False) 
        # print('--- top latency ---') 
        # print_list(utilities[:20])
        # utilities.sort(key=lambda x: x[2], reverse=False) 
        # print('--- top accuracy ---')
        # print_list(utilities[:20]) 

        utilities.sort(key=lambda x: x[0], reverse=True)
        return utilities[0], 0
    
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
                util, lat = self.calculate_utility(result, placement)
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

                util, lat = self.calculate_utility(result, placement) 
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

                    util, lat = self.calculate_utility(result, placement) 
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
        
if __name__ == "__main__":
   
    result = []
    def run(method, num_trails, min_accuracy, max_latency, num_profile_sample):
        engine = Engine(min_accuracy=min_accuracy, max_latency=max_latency, num_profile_sample=num_profile_sample)

        num_qualified_trail = 0
        cache_size = []
        best_utils = []
        profile_overheads = []
        for _ in range(num_trails):
            if method == 'all bo':
                placement, profile_lat = engine.all_bo()
            elif method == 'vulcan': 
                placement, profile_lat = engine.select_best_placement()
            elif method == 'vulcan+acc_prior':
                placement, profile_lat = engine.select_best_placement(optimize=True)
            elif method == 'random': 
                placement, profile_lat = engine.random_iterative(50)
            elif method == 'acc_lat_bo_const':
                placement, profile_lat = engine.experiment2()
            elif method == 'exhaustive':
                placement, profile_lat = engine.exhaustive_search_best_placement()
                print(placement)
            else: 
                raise KeyError('Unsupported key')
                return 
            # print(placement)
            # print(len(engine.cache))
            cache_size.append(len(engine.cache))
            best_utils.append(placement[0])
            profile_overheads.append(profile_lat)
            print(profile_lat)
            if placement[1] <=  max_latency and placement[2] <= min_accuracy: 
                num_qualified_trail += 1
            
            print(placement)
        result.append( dict(
            optimization_type=method,
            min_accurac=min_accuracy,
            max_latency=max_latency, 
            num_profile_sample=num_profile_sample,
            num_trails=num_trails,
            qualify_optimization=num_qualified_trail,
            cache_size=cache_size,
            best_utils=best_utils, 
            profile_overheads=profile_overheads,
        ))
        print(f'{method} \
                #qualify optimization: {num_qualified_trail}/{num_trails} \
                avg cache size: {sum(cache_size)/num_trails} \
                avg utility: {sum(best_utils)/num_trails}' )
    
        
    # 100, 300, 500, 1000, 2000, 3000, 4000, 5000,  
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=1)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=300)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=500)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=1000)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=2000)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=3000)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=4000)
    run('vulcan', num_trails=5, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    run('exhaustive',  num_trails=1, min_accuracy=0.35, max_latency=0.2, num_profile_sample=5000)
    # run('acc_lat_bo_const')

    with open('opt_result.json', 'w') as fp:
        json.dump(result, fp)=