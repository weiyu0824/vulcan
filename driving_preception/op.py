from nuimages import NuImages
import tqdm
import cv2
import numpy as np
from ultralytics import YOLO 
import torch
from PIL import Image 
from numpy import ndarray

from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from base_op import ProcessOp, SourceOp

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

class Loader(SourceOp):
    def __init__(self, args: dict):
        pass

class GroundRemover(ProcessOp):
    def __init__(self):
        pass

class Voxelization(ProcessOp):
    def __init__(self):
        pass

class ThreeDimDetector(ProcessOp):
    def __init__(self):
        pass
