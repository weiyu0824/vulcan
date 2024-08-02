import os
import random
from torch.utils.data import Dataset
UNIT_LEN = 1000

class Sampler():
    def __init__(self) -> None:
        pass
    def init(self):
        pass
    def sample(self, ctx):
        pass
    def feedback(self, ctx):
        pass
    def update_weight():
        pass
    
class NuImagesRandomSampler(Sampler):
    def __init__(self, dataset: Dataset):
        self.random_idx = 0
        self.ds = dataset
        self.n = len(dataset)
        self.keys = list(range(self.n))
        
    def init(self):
        self.random_idx = 0
        random.shuffle(self.keys)
        
    def random_sample(self):
        idx = self.random_idx
        self.random_idx = (self.random_idx + 1) % len(self.keys)
        return self.keys[idx]
    

    def sample(self, method):
        if method == "random":
            return self.random_sample()
        else:
            raise Exception(f"Invalid sample method [{method}]")
        
class NuImagesStratifiedSampler(Sampler):
    def __init__(self, dataset, cluster):
        self.ds = dataset
        self.keys = list(cluster.keys()).copy()
        self.k2g = cluster.copy()
        self.n_groups = len(set(list(cluster.values())))
        self.next_group = 0
        self.group = [
            [k for k in self.keys if self.k2g[k] == i] for i in range(self.n_groups)
        ]
        self.group_idx = [0] * self.n_groups
        res = ""
        for i in self.group:
            res = res + " " + str(len(i))
        print(f"Weighted Sampler: {res}")
        
    def init(self):        
        self.next_group = 0
        self.group_idx = [0] * self.n_groups
        for i in range(self.n_groups):
            random.shuffle(self.group[i])
            
        random.shuffle(self.keys)
       
    def sample_group(self):
        key = random.choice(self.keys)
        return self.k2g[key]
    
    def stratified_sample(self):
        g = self.sample_group()
        
        idx = self.group_idx[g]
        self.group_idx[g] = (self.group_idx[g] + 1) % len(self.group[g])
        key = self.group[g][idx]
        
        return key
    
    def sample(self, method):
        if method == "stratified":
            return self.stratified_sample()
        else:
            raise Exception(f"Invalid sample method [{method}]")
           