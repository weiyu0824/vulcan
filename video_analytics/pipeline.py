from base_op import ProcessOp, SourceOp
from typing import List
import numpy as np
from nuimages import NuImages
import numpy as np
from ultralytics import YOLO 
import torch
from metric import calculate_map
from PIL import Image 
from utils import get_dict_coco_category_to_id, get_dict_nuimage_category_to_id
from numpy import ndarray
from utils import num_classes 
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from base_op import ProcessOp, SourceOp


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


class MeanAveragePrecisionEvaluator(Evaluator):
    def __init__(self):
        super().__init__("map")
    
    def evaluate(self, y, label):
        instance = MeanAveragePrecision()
        instance.update(y, label)
        res = instance.compute()['map_50'].item()
        return res


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
        result = self.ev.evaluate(x["preds"], x["targets"])
        self.results.append(result)
        return result, x
    
    def run_cached(self, name):
        return self.cache["results"][name]
        
        
        
    