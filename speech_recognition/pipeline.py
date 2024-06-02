from base_op import ProcessOp, SourceOp
from typing import List

class BatchData():
    def __init__(data, label, name: str):
        self.data = data
        self.label = label
        self.name = name

class Evaluator():
    def __init__(name: str):
        self.name = name
    
    def evaluate(y, label):
        raise NotImplementedError

class werEvaluator():
    def __init__():
        super().__init__("wer")
    
    def evaluate(y, label):
        return jiwer.wer(label, y)

class Pipeline():
    def __init__(name: str, ops: List[ProcessOp], evalulator: Evaluator, cache: Dict):
        self.ops = ops
        self.name = name
        self.evaluator = evaluator 
        self.results = []
        self.cache = cache
    
    def run(batch: BatchData):
        x = batch.data
        for op in ops:
            x = op.process(x)
        result = self.evaluator.evaluate(y, batch.label)
        self.results.append(result)
        return result
    
    def run_cached(x):
        x = batch.name
        return cache[x]
        
        
        
    