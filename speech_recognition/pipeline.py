from base_op import ProcessOp, SourceOp
from typing import List
import numpy as np
import torch
import jiwer
import pickle
import torchaudio
import time
from torchgating import TorchGating as TG

class BatchData():
    def __init__(self, data, label, name: str):
        self.data = data
        self.label = label
        self.name = name

class Evaluator():
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, y, label):
        raise NotImplementedError


class WordErrorRate(Evaluator):
    def __init__(self):
        super().__init__("wer")
    
    def evaluate(self, y, label):
        return jiwer.wer(label, y)


class Pipeline():
    def __init__(self, name: str, ops: List[ProcessOp], evaluator: Evaluator, cache):
        self.ops = ops
        self.name = name
        self.ev = evaluator 
        self.results = []
        self.cache = cache
        
    def run(self, batch: BatchData):
        x = batch.data
        for op in self.ops:
            x = op.process(x)
        result = self.ev.evaluate(x["prediction"], batch.label)
        self.results.append(result)
        return result, x
    
    def run_cached(self, name):
        return self.cache["results"][name]["metric"]
        
        
        
    