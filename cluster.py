from data import TierSpec, NodeState, Query
from typing import List



class ClusterState:
    INIT_SCORE = -1000000

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
        self.all_quries_latencies_ok = True 

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
            # print(deployed_query['query'].lat_thold, pipe_latency) 
            if deployed_query['query'].lat_thold < pipe_latency:
                self.all_quries_latencies_ok = False

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

    def queries_meet_latencies(self):
        return self.all_quries_latencies_ok

    def exceed_score(self, score_key: str, score):
        if self.is_cluster_out_of_memory() or self.queries_meet_latencies() == False:
            return False 
        else:
            if score == INIT_SCORE:
                return True
            if ((score_key == 'utility' and self.tot_util > score) or
                (score_key == 'gpu_time' and self.tot_gpu_processing_time < score)):
                return True
            return False 
    def get_score(self, score_key: str):
        if score_key == 'utility': return self.tot_util
        elif score_key == 'gpu_time': return self.tot_gpu_processing_time
        else: raise Exception('Unsupported score key')
