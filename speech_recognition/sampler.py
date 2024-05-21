import os
import random

class Sampler():
    def __init__(self) -> None:
        pass
    def sample(self, ctx):
        pass

class VOiCERandomSampler(Sampler):
    def __init__(self) -> None:
        self.random_data = [] 
        self.random_idx = 0
        self.stratified_data = dict()
        self.stratified_weight = dict()
        self.stratified_idx = dict()
        self.get_all_data()
        print(f"Summary:\n\t#Samples:{len(self.random_data)}")
         
    def get_all_data(self):
        l0 = "/data/VOiCES_devkit/distant-16k/speech/test"
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        for l1 in l1s:
            self.stratified_data[l1] = dict()
            self.stratified_weight[l1] = dict()
            self.stratified_idx[l1] = dict()
            for l2 in l2s:
                self.stratified_data[l1][l2] = []
                self.stratified_weight[l1][l2] = 0
                self.stratified_idx[l1][l2] = 0
                
        for l1 in l1s:
            for l2 in l2s:
                path = l0 + '/' + l1 + '/' + l2
                sps = [f"{path}/{sp}" for sp in os.listdir(path) if sp.startswith('sp')]
                for sp in sps:
                    fs = os.listdir(sp)
                    for f in fs:
                        self.random_data.append(f"{sp}/{f}")
                        self.stratified_data[l1][l2].append(f"{sp}/{f}")
                random.shuffle(self.stratified_data[l1][l2])
        culmulative_weight = 0
        for l1 in l1s:
            for l2 in l2s:
                culmulative_weight += len(self.stratified_data[l1][l2]) / len(self.random_data)
                self.stratified_weight[l1][l2] = culmulative_weight
                self.stratified_idx[l1][l2] = 0
        random.shuffle(self.random_data)

    
    def random_sample(self):
        idx = self.random_idx
        self.random_idx = (self.random_idx + 1) % len(self.random_data)
        return self.random_data[idx]

    def sample_strata_based_on_weight(self):
        weight = random.uniform(0, 1)
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        for l1 in l1s:
            for l2 in l2s:
                if len(self.stratified_data[l1][l2]) > 0 and weight < self.stratified_weight[l1][l2]:
                    return l1, l2
        raise Exception("Weight not in range")
    
    def stratified_sample(self):  
        l1, l2 = self.sample_strata_based_on_weight()
        idx = self.stratified_idx[l1][l2]
        data = self.stratified_data[l1][l2]
        self.stratified_idx[l1][l2] = (idx + 1) % len(data)
        return data[idx] 

    def sample(self, ctx) -> str:
        if ctx == "random":
            return self.random_sample()
        elif ctx == "stratified":
            return self.stratified_sample()

