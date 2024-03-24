from bo import BayesianOptimization

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
        self.n_init_samples = 3


        self.pipeline = Pipeline('video_analytics')
        self.machines = [{'address': 'localhost'}]



    def load_data_from_cache(config) -> ExecutionStats:
        pass

    def run_pipeline(self, placement, config):
        # hack

        # Send docker images to different machines 

        # Maintain TCP connection


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
        for knob in knobs: 
            value = knob['value']
            if knob['type'] == 'Range':
                cont_bounds.append([value[0], value[1]])        
            elif knob['type'] == 'Category':
                cat_bounds.append([i for i in range(len(value))])

        placements = self.generate_all_placement_choices()
        utilities = []
        for placement in placements:
            # BO
            bo = BayesianOptimization(cont_bounds, cat_bounds=cat_bounds) 
            init_utilities = [] # utility

            # init BO 
            # 1. Random sample configs
            init_samples = bo.random_sample(n_samples=self.n_init_samples)
            print('init sample config', init_samples)
            # 2. Run those config to get ACC & Latency
            for init_sample in init_samples:
                result = self.run_pipeline(placement=placement, config=init_sample)
                utility = self.calculate_utility(result)
                init_utilities.append(utility)
            # 3. Fit those samples into BO 
            bo.fit(X=init_samples, y=init_utilities)

            prune_counter = 0  
            prev_utility = 0 
            while True:
                config = bo.get_next_sample() 
                print('explore config', config)

                # OPT: cache 'accuracy' result, and only meausure 'latency'  
                result = self.run_pipeline(config, placement)  
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


engine = Engine()
placement = engine.select_best_placement()