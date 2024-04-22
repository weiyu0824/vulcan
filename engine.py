from bo import BayesianOptimization
import docker 
import socket
import json 

target_accuracy = 0.5
target_latency = 3.0
perf_weight = 0.5
# hyper-params:
accuracy_coef = 1
latency_coef = 1
compute_coef = 1
network_coef = 1
# g:
# aA:
# aL:
# agpu:
# anet:
from typing import Tuple, Dict, List
from pipeline import Pipeline


video_analytics_knobs = [
    ('models', ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']),
    ('resize_factor', [0.9, 0.8, 0.7, 0.6, 0.5])
]
def sample_to_configs(sample: List[int], knob_list)-> dict: 
    config = {}
    assert len(sample) == len(knob_list), 'Sample and should have equal size as knob'

    for knob_idx, (knob_name, knob_choices) in enumerate(knob_list): 
        config[knob_name] = knob_choices[sample[knob_idx]]
    return config

class MachineInfo:
    address: Tuple[str, int]
    resouce: str # cpu or gpu 
    level: int # order 

class ExecutionStats:
    def __init__(self, accuracy, latency):
        self.accuracy: float = accuracy 
        self.latency: float = latency
        # self.compute_cost: float = compute_cost
        # self.network_cost: float = network_cost


class Engine:
    def __init__(self, min_accuracy, max_latency):
        self.early_prune_count = 3
        self.bo_stop_count = 5
        self.num_init_sample_configs = 3
        self.utility_thold = 0  

        self.min_accuracy = min_accuracy
        self.max_latency = max_latency

        # self.pipeline = Pipeline('video_analytics')
        self.machines = [{'address': 'localhost'}]

        # self.docker_client = docker.from_env()

        self.port = 12343
        self.host = "localhost"

        self.knob_list = video_analytics_knobs

    # def load_data_from_cache(config) -> ExecutionStats:
    #     pass

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
        print('config: ', config)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Bind the socket to the address and port
            server_socket.bind((self.host, self.port))

            # Listen for incoming connections
            server_socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            # Open the container
            container = self.docker_client.containers.run(
                image='video-task',
                detach=True,  # Detach container so it runs in the background
                environment=config,  # Pass config as environment variables
                network_mode='host',  # Use host network for TCP connection
                volumes={
                    '/dev/shm/wylin/nuimages': {'bind': '/nuimages', 'mode': 'rw'},
                    '/users/wylin2/vulcan/video_analytics': {'bind': '/app', 'mode': 'rw'}
                },
                auto_remove=True,
                runtime='nvidia',  # Specify NVIDIA runtime for GPU support                
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

    def cache_pipeline_results():
        pass

    def calculate_utility(self, profile_stats, placement) -> float:
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

        mbs_per_sec = 20
        cpu_gpu_processing_diff = 25
        appox_latency = 0
        appox_latency += profile_stats['loader_compute_latency'] 
        if placement == [0, 1]: # edge -> cloud
            appox_latency += profile_stats['loader_output_size'] / (1e6 * mbs_per_sec)
            appox_latency += profile_stats['detector_compute_latency']
        else: # edge -> edge
            appox_latency += profile_stats['detector_compute_latency'] * cpu_gpu_processing_diff 

        utility = 0.5 * (profile_stats['accuracy'] - self.min_accuracy) + 0.5 * (self.max_latency - appox_latency)

        return utility, appox_latency

    def generate_all_placement_choices(self):

        # hack
        return [[0, 0], [0, 1]]



    def select_best_placement(self):

        feat_bounds = []
        for _, knob_choice in self.knob_list:
            feat_bounds.append([i for i in range(len(knob_choice))])
        

        placements = self.generate_all_placement_choices()
        utilities = []
        for placement in placements:
            # BO
            bo = BayesianOptimization(feat_bounds=feat_bounds) 
            init_utilities = [] # utility

            # init BO 
            # 1. Random sample configs
            init_sampels = bo.get_random_samples(num_samples=self.num_init_sample_configs)
            # 2. Run those config to get ACC & Latency
            for sample in init_sampels:
                
                # Convert BO sample back to config
                config = sample_to_configs(sample, self.knob_list)
                
                result = self.run_pipeline(config=config)
                utility, appox_latency = self.calculate_utility(result, placement)
                init_utilities.append(utility)

                utilities.append((utility, appox_latency, config, placement, result))

            # 3. Fit those samples into BO 
            bo.fit(X=init_sampels, y=init_utilities)

            # BO optimization 
            prune_count = 0  
            prev_utility = max(init_utilities)
            attempt_idx = 0 
            stop_count = 0
            while True:
                print('Attempt: ', attempt_idx, 'stop cnt: ', stop_count, 'prune cnt: ', prune_count, 'prev_utility: ', prev_utility)
                attempt_idx += 1
                sample = bo.get_next_sample()
                env_vars = sample_to_configs(sample, self.knob_list) 

                # OPT: cache 'accuracy' result, and only meausure 'latency'  
                result = self.run_pipeline(config=env_vars)  
                utility, appox_latency = self.calculate_utility(result, placement) 

                # threshold = min utility to meet requirement = 0,
                # since (Am-Am) + (Lm-Lm) = 0 
                # 1. Early stop
                
                if utility <= self.utility_thold:
                    prune_count += 1
                    if prune_count >= self.early_prune_count:
                        break
                
                utilities.append((utility, appox_latency, env_vars, placement, result))

                # 2. BO stop
                if utility < prev_utility * 1.1:
                    stop_count += 1
                else:
                    stop_count = 0 
                if stop_count >= self.bo_stop_count:
                    break
                prev_utility = utility
                print()

        utilities.sort(key=lambda x: x[0])
        return utilities[len(utilities) - 1]

if __name__ == "__main__":

    engine = Engine(min_accuracy=0.4, max_latency=3)
    placement = engine.select_best_placement()
    print(placement)
