from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Operator:
    name: str
    tpy: str # basic or dnn

@dataclass
class Knob: 
    name: str
    choice: list 

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

@dataclass
class SetupMetric:
    accuracy: float 
    latency: float
    utility: float
    cost: float
    tot_transfer_latency: float
    tot_gpu_time: float
    accuracy_index: float
    compute_latency_by_op: list[float]
    gpu_mem_usg_by_op: list[float] 

@dataclass
class QuerySetup:
    query: Query
    placement: List[int]
    config: Dict
    setup_metric: SetupMetric

