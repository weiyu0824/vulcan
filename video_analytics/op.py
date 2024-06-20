from nuimages import NuImages
import tqdm
import cv2
import numpy as np
from ultralytics import YOLO 
import torch
from metric import calculate_map
from PIL import Image 
from utils import get_dict_coco_category_to_id, get_dict_nuimage_category_to_id
from numpy import ndarray
import pickle
import time
import json
from utils import num_classes 
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from base_op import ProcessOp, SourceOp

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

data_root = '/data/wylin2/nuimages/'
data_version = 'v1.0-mini'

class Loader(SourceOp):
    def __init__(self):
        # self.root = '/nuimages/'
        self.root = data_root 
        self.version = data_version 
        
        self.nuim = NuImages(dataroot=self.root, version=self.version, verbose=False, lazy=True)
        self.num_samples = len(self.nuim.sample)
        # print(self.num_samples)
        # print(self.nuim.sample[0])
        # exit()
        # self.batch_size = 1 
        # self.resize_shape = args['resize_shape']

        # self.cur_batch_idx = 0
        self.nuim_category_to_id_dict = get_dict_nuimage_category_to_id()

        self.output_data_size = None
        self.compute_latencies = []

        # with open(f'{self.version}-idx.json', 'r') as fp:
        #     self.index_data = json.load(fp)

    def save_index(self):
        with open(f'{self.version}-idx.json', 'w') as fp:
            json.dump(self.index_data, fp) 

    def build_index(self):
        # start_idx = 0
        batch_imgs = []
        batch_labels = []
        start_compute_time = time.time()
        org_image_shape = None


        print(self.nuim.sample[0])

        # cur_idx = start_idx
        # while (cur_idx < min(self.num_samples, start_idx + self.batch_size)):
        self.index_data = [] 
        for cur_idx in tqdm.tqdm(range(0, self.num_samples)):
            sample = self.nuim.sample[cur_idx]
            img_fname = self.nuim.get('sample_data', sample['key_camera_token'])['filename']
            # img_fname = self.index_data[cur_idx]['img_fname']
            
            img_source = cv2.imread(self.root + img_fname)
            org_image_shape = img_source.shape  
            
            # img_source = cv2.resize(img_source, (self.resize_shape, self.resize_shape)) # (900, 1600)
            # img_source: ndarray = img_source.astype(np.float32) / 255.0
             
            # batch_imgs.append(img_source/nuimages)
            true_boxes = self.get_ground_truth_box(sample).tolist()
            # true_boxes = self.index_data[cur_idx]["true_boxes"]
            batch_labels.append(true_boxes)

            # index
            self.index_data.append({
                "img_fname": img_fname,
                "true_boxes": true_boxes
            })
        self.save_index()

    def get_ground_truth_box(self, sample):
        if not sample:
            raise ValueError('sample should not be none')
        key_camera_token = sample['key_camera_token']
        # object_tokens, surface_tokens = nuim.list_anns(sample['token'])
        bboxes = [o['bbox'] for o in self.nuim.object_ann if o['sample_data_token'] == key_camera_token]
        categories = [self.nuim.get('category', o['category_token'])['name']
                    for o in self.nuim.object_ann if o['sample_data_token'] == key_camera_token]
        # print(categories)
        # print(bboxes)    

        # num_boxes = len(bboxes)
        # categories = torch.zeros(size=(len(categories), 1))
        # bboxes = torch.tensor(bboxes)
        # print('gt' , categories)
        filtered_categories, filtered_bboxes = [], []
        # true_boxes = []
        for category, bbox in zip(categories, bboxes):
            # print(type(category), category)
            if category not in self.nuim_category_to_id_dict:
                # print('filter: ', category)
                continue
            filtered_categories.append(self.nuim_category_to_id_dict[category])
            filtered_bboxes.append(bbox)
            # true_boxes.append(torch.cat([torch.tensor(bbox), torch.tensor(NuimageCategoryToId[category])]))
        # print(true_boxes)
        
        filtered_bboxes = torch.tensor(filtered_bboxes)
        filtered_categories = torch.unsqueeze(torch.tensor(filtered_categories), dim=1) 
        true_boxes = torch.cat([filtered_bboxes, filtered_categories], dim=1) 
        # print('true', true_boxes)
        # exit(0)
        return true_boxes 

    def get_output_data_size(self):
        """Output datasize per datapoint"""
        return self.output_data_size

    def get_compute_latency(self):
        """Compute latency per batch"""
        return sum(self.compute_latencies)/len(self.compute_latencies)

class NuImageDataset(Dataset):
    def __init__(self):
        self.version = data_version 
        self.root = data_root 
        with open(f'{self.version}-idx.json', 'r') as fp:
            self.index_data = json.load(fp)
        self.max_num_bbox = 0

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, index):
        img_fname = self.index_data[index]['img_fname']
        img_source = cv2.imread(self.root + img_fname)

        # img_source = cv2.resize(img_source, (640, 640)) # (900, 1600)
        # img_source: ndarray = img_source.astype(np.float32) / 255.0
        img_source = np.transpose(img_source, (2, 0, 1))

        # boxes
        padded_boxes = self.index_data[index]["true_boxes"]

        for _ in range(50 - len(padded_boxes)):
            padded_boxes.append([0, 0, 0, 0, -1])
        
        padded_boxes = torch.tensor(padded_boxes)

        return img_source, padded_boxes

class GroundRemoval(ProcessOp):
    def __init__(self):
        pass

class Voxelization(ProcessOp):
    def __init__(self, org_image_shape, args: dict):
        super().__init__()
        w = int(args['resize_factor'] * org_image_shape[0])
        h = int(args['resize_factor'] * org_image_shape[1])
        print(w, h)
        self.resize = Resize(size=(h, w))

    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data['images']))

        if profile_compute_latency:
            start_compute_time = time.time()

        # 1. to device 2. normalize 3. resize
        images = batch_data['images'].to(device)
        images = self.resize(images)

        if profile_compute_latency:
            self.compute_latencies.append(time.time() - start_compute_time)

        batch_data['images'] = images
        return batch_data

        

class Detector(ProcessOp):
    def __init__(self, org_image_shape: tuple, args: dict):
        super().__init__()
        # Knobs
        model_name = args['model_name']
        self.model = YOLO(f'{model_name}.pt')
        self.coco_category_to_id_dict = get_dict_coco_category_to_id()
        self.compute_latencies = [] 
        self.batch_accuraies = []
        self.resize = Resize(size=(args['resize_shape'], args['resize_shape']))
        h, w = org_image_shape[0], org_image_shape[1]
        self.resize_matrix = torch.tensor([
            [w, 0, 0, 0],
            [0, h, 0, 0],
            [0, 0, w, 0],
            [0, 0, 0, h]
        ], dtype=torch.float32).to(device)
        self.metric = MeanAveragePrecision()
    
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data['images']))
        
        if profile_compute_latency:
            start_compute_time = time.time()
        
        # compute
        labels = batch_data['labels']
        # org_image_shape = batch_data['org_image_shape']
        # batch_tensor = torch.tensor(np.array(batch_data['images']), dtype=torch.float32)

        # t = time.time()
        batch_tensor = (batch_data['images'] / 255.0).to(device)
        # print('/ ', time.time() - t)
        # t = time.time()
        batch_tensor = self.resize(batch_tensor)
        # print('r ', time.time() - t)
        results = self.model.predict(batch_tensor, save=False, verbose=False, classes=[0, 1, 2, 3, 5, 7])
        
        if profile_compute_latency:
            self.compute_latencies.append(time.time()-start_compute_time)

        

        preds, targets = [], []
        for result, label in zip(results, labels):

            # skip this frame if there isn't any annotation.
            if label[0][4] == -1:
                continue
            
            # target
            target_boxes, target_labels = [], []
            for box in label:
                true_cls = box[4]
                if true_cls == -1:
                    break
                else:
                    target_boxes.append(box[:4])
                    target_labels.append(box[4])
            targets.append(dict(
                boxes=torch.stack(target_boxes),
                labels=torch.stack(target_labels),
            ))

            # pred 
            raw_pred_clss = result.boxes.cls.cpu()  
            raw_pred_confs = result.boxes.conf.cpu()
            raw_pred_bboxes = result.boxes.xyxyn
            raw_pred_bboxes = torch.matmul(raw_pred_bboxes, self.resize_matrix).round().int().cpu()
            # filter unwanted boxes
            pred_boxes, pred_scores, pred_labels = [], [], []
            for pred_cls, pred_conf, pred_bbox in zip(raw_pred_clss, raw_pred_confs, raw_pred_bboxes):
                pred_bbox = pred_bbox.tolist()
                pred_cls = int(pred_cls.item())
                if pred_cls not in self.coco_category_to_id_dict:
                    continue

                pred_labels.append(self.coco_category_to_id_dict[pred_cls])  
                pred_scores.append(pred_conf)
                pred_boxes.append(pred_bbox)

            preds.append(dict(
                boxes=torch.tensor(pred_boxes),
                scores=torch.tensor(pred_scores),
                labels=torch.tensor(pred_labels)
            ))


            
               
        # metric = MeanAveragePrecision()
        self.metric.update(preds, targets)
        # map_50 = metric.compute()['map_50']
            # mean_AP = calculate_map(true_boxes=true_boxes, pred_boxes=pred_boxes, iou_threshold=0.5, num_classes=num_classes)
            
            
        # print(map_50)      
            
            
        # exit()
        # mean_APs.append(mean_AP.item()) 

        # self.batch_accuraies.append(sum(mean_APs)/len(mean_APs)) 
        return 
    
    def record_preds(self, results):

        pass

    def get_compute_latency(self):
        return sum(self.compute_latencies)/len(self.compute_latencies)

    def get_endpoint_accuracy(self):
        # if len(self.batch_accuraies)  == 0:
        #     return 0
        # return sum(self.batch_accuraies)/len(self.batch_accuraies)
        batch_accuracy = self.metric.compute()['map_50'].item()
        self.batch_accuraies.append(batch_accuracy)
        # clear metric tape
        self.metric = MeanAveragePrecision()
        return sum(self.batch_accuraies) / len(self.batch_accuraies) 
    
    def get_batch_accuracy(self):
        return self.batch_accuraies
