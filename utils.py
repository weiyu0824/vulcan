from scipy.stats import norm
from typing import List
from data import Knob, Query


def probability_greater_than_t(mu, sigma, t):
    z = (t - mu) / sigma
    cdf_value = norm.cdf(z)
    probability = 1 - cdf_value
    return probability

def sample_to_config(sample: List[int], knob_list: List[Knob])-> dict: 
    config = {}
    assert len(sample) == len(knob_list), "Sample and knob should have equal size as "

    for knob_idx, knob in enumerate(knob_list): 
        config[knob.name] = knob.choice[sample[knob_idx]]
    return config

def config_to_sample(config: dict, knob_list: List[Knob]) -> List[int]:
    sample = []
    for knob in knob_list:  
        i = knob.choice.index(config[knob.name])
        sample.append(i)
    return sample

def get_feat_bounds(query: Query):
    feat_bounds = []
    for knob in query.knob_list: 
        feat_bounds.append([i for i in range(len(knob.choice))]) 
    
    return feat_bounds