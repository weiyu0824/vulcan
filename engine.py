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

class MachineInfo:
    address: Tuple[str, int]
    resouce: str # cpu or gpu 
    level: int # order 

class ExecutionStats:
    def __init__(self, accuracy, latency, compute_cost, network_cost):
        self.accuracy: float = accuracy 
        self.latency: float = latency
        self.compute_cost: float = compute_cost
        self.network_cost: float = network_cost


class Engine:
    def __init__(self):
        self.early_prune_count = 3
        self.bo_stop_count = 5
        self.num_init_sample_configs = 3


        self.pipeline = Pipeline('video_analytics')
        self.machines = [{'address': 'localhost'}]

        self.docker_client = docker.from_env()

    def load_data_from_cache(config) -> ExecutionStats:
        pass

    def get_execution_stats(self, container_addr='localhost', container_port=12345):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                
                s.settimeout(5) 
                print('start connecnting...')

                s.connect((container_addr, container_port))  # Example TCP port
                # Send request for stats
                s.sendall(b"get_stats")
                # Receive response
                data = s.recv(1024)
                # Parse JSON data
                stats = json.loads(data.decode())
                # Extract stats
                cpu_usage = stats['cpu_usage']
                # Create ExecutionStats object
                print('recv result')
                return ExecutionStats(cpu_usage, 0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            print(f"Error: {e}")
            # Return default stats
            return ExecutionStats(0.0, 0.0, 0.0, 0.0)

    def run_pipeline(self, config: dict):
        """
        Run the pipeline to get 1. the output_size of each operator & 2. overall accuracy
        """

        # hack

        # Send docker images to different machines -> (put entire pipeline in 1 image)
        # print('config', config)
        # exit()

        # Maintain TCP connection
        container = self.docker_client.containers.run(
            image='video-task',
            detach=True,  # Detach container so it runs in the background
            environment=config,  # Pass config as environment variables
            network_mode='host',  # Use host network for TCP connection
            # ports={'12345/tcp': 12345} 
        )

        # Get container ID
        container_id = container.id

        # Maintain TCP connection to the container and wait for return stats
        # res = self.get_execution_stats(container_id)

        # # Stop and remove the container
        # container.stop()
        # container.remove()
        res = self.get_execution_stats()

        exit(0)

        return res
        
        


        res = ExecutionStats(0.5, 0.5, 0.4, 0.2)
        return res 

    def cache_pipeline_results():
        pass

    def calculate_utility(self, exec_stat: ExecutionStats) -> float:
        # Uq,p,c = Pq,p,c/Rq,p,c
        # Pq,p,c(A,L) = g·aA ·(A - Am)+(1 - g)·aL ·(Lm - L)
        #             = weight * Accuracy + weight * Latency
        # Rq,p,c = agpu · Rgpu + anet · Rnet
        #        = gpu processing time + consumed network bandwidth
        
        perf = perf_weight * exec_stat.accuracy + (1 - perf_weight) * exec_stat.latency 
        resource = compute_coef * exec_stat.compute_cost + network_coef * exec_stat.network_cost
        utility = perf / resource
        return utility 

    def generate_all_placement_choices(self):

        # hack
        return [[0, 0], [1, 1], [0, 1]]


    def select_best_placement(self):

        # Pipeline knobs 
        knobs = self.pipeline.get_knobs()
        cont_bounds, cat_bounds = [], []
        knob_map = {}
        for knob in knobs: 
            value = knob['value']
            if knob['type'] == 'Range':
                knob_map[f'range-{len(cont_bounds)}'] = knob    
                cont_bounds.append([value[0], value[1]]) 
            elif knob['type'] == 'Category':
                knob_map[f'category-{len(cat_bounds)}'] = knob  
                cat_bounds.append([i for i in range(len(value))])

        placements = self.generate_all_placement_choices()
        utilities = []
        for placement in placements:
            # BO
            bo = BayesianOptimization(cont_bounds=cont_bounds, cat_bounds=cat_bounds) 
            init_utilities = [] # utility

            # init BO 
            # 1. Random sample configs
            init_configs = bo.get_random_samples(num_samples=self.num_init_sample_configs)
            print('init sample config', init_configs)
            # 2. Run those config to get ACC & Latency
            for config in init_configs:
                
                # Convert config back to env vars
                env_vars = {}
                for i, feat in enumerate(config['cont_feats']):
                    env_vars[knob_map[f'range-{i}']['name']] = feat
                for i, feat in enumerate(config['cat_feats']):
                    knob = knob_map[f'category-{i}']
                    env_vars[knob['name']] = knob['value'][int(feat)] 


                result = self.run_pipeline(config=env_vars)
                utility = self.calculate_utility(result)
                init_utilities.append(utility)
            # 3. Fit those samples into BO 
            bo.fit(X=init_configs, y=init_utilities)

            # BO optimization 
            prune_counter = 0  
            prev_utility = 0 
            while True:
                config = bo.get_next_sample() 
                print('explore config', config)

                # OPT: cache 'accuracy' result, and only meausure 'latency'  
                result = self.run_pipeline(config)  
                utility = self.calculate_utility(result) 

                # threshold = min utility to meet requirement = 0,
                # since (Am-Am) + (Lm-Lm) = 0 
                # 1. Early stop
                utility_thold = 0  
                if utility <= utility_thold:
                    prune_counter += 1
                    if prune_counter >= self.early_prune_count:
                        break
                
                utilities.append((utility, config, placement))

                # 2. BO stop
                if utility < prev_utility * 1.1:
                    stop_count += 1
                else:
                    stop_count = 0 
                if stop_count >= self.bo_stop_count:
                    break
                prev_utility = utility
            
        utilities.sort(key=lambda x: x[0])

        return utilities[0]

if __name__ == "__main__":
    # client = docker.from_env()
    # exit()
    engine = Engine()
    placement = engine.select_best_placement()