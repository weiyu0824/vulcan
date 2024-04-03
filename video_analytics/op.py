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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
class Op:
    def __init__(self):
        pass

    def get_result(self):
        NotImplementedError()

class ProcessOp(Op):
    def __init__(self):
        self.compute_latencies = [] 
        self.input_size = None

    def profile(self, batch_data):
        """Do some transformation"""
        NotImplementedError()

    def get_compute_latency(self):
        if len(self.compute_latencies) == 0:
            return 0
        return sum(self.compute_latencies)/len(self.compute_latencies)

    def get_input_size(self):
        return self.input_size 

class SourceOp(Op):
    def load_batch(self):
        """Do some transformation"""
        NotImplementedError() 

class Loader(SourceOp):
    def __init__(self, args: dict):
        # self.root = '/nuimages/'
        self.root = '/dev/shm/wylin/nuimages/'
        self.version = 'v1.0-val'
        
        self.nuim = NuImages(dataroot=self.root, version=self.version, verbose=False, lazy=True)
        self.num_samples = len(self.nuim.sample)
        self.batch_size = args['batch_size']
        self.resize_shape = args['resize_shape']

        self.cur_batch_idx = 0
        self.nuim_category_to_id_dict = get_dict_nuimage_category_to_id()

        self.output_data_size = None
        self.compute_latencies = []

        with open(f'{self.version}-idx.json', 'r') as fp:
            self.index_data = json.load(fp)

    def save_index(self):
        with open(f'{self.version}-idx.json', 'w') as fp:
            json.dump(self.index_data, fp) 

    def load_batch(self):
        start_idx = self.cur_batch_idx * self.batch_size
        batch_imgs = []
        batch_labels = []
        start_compute_time = time.time()
        org_image_shape = None
        cur_idx = start_idx
        while (cur_idx < min(self.num_samples, start_idx + self.batch_size)):
            sample = self.nuim.sample[cur_idx]
            # img_fname = self.nuim.get('sample_data', sample['key_camera_token'])['filename']
            img_fname = self.index_data[cur_idx]['img_fname']
            
            img_source = cv2.imread(self.root + img_fname)
            org_image_shape = img_source.shape  
            
            img_source = cv2.resize(img_source, (self.resize_shape, self.resize_shape)) # (900, 1600)
            img_source: ndarray = img_source.astype(np.float32) / 255.0
            
            batch_imgs.append(img_source)
            # true_boxes = self.get_ground_truth_box(sample).tolist()
            true_boxes = self.index_data[cur_idx]["true_boxes"]
            batch_labels.append(true_boxes)

            # index
            # self.index_data.append({
            #     "img_fname": img_fname,
            #     "true_boxes": true_boxes
            # })

            cur_idx += 1
        
        # Transformation 
        batch_tensor = np.array(batch_imgs)
        batch_tensor = np.transpose(batch_tensor, (0, 3, 1, 2))

        batch_data = {
            'images': batch_tensor,
            'labels': batch_labels, 
            'org_image_shape': org_image_shape
        }
        self.compute_latencies.append(time.time() - start_compute_time)


        # Record the data_size
        if (self.output_data_size == None):
            self.output_data_size = len(pickle.dumps(batch_data))/self.batch_size

        # Update batch_id
        self.cur_batch_idx += 1

        return batch_data

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
        self.version = 'v1.0-val'
        self.root = '/dev/shm/wylin/nuimages/' 
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

        # if (len(true_boxes) > self.max_num_bbox):
        #     self.max_num_bbox = len(true_boxes)
        #     print('gogogs', self.max_num_bbox)
        return img_source, padded_boxes

class GroundRemoval(ProcessOp):
    def __init__(self):
        pass

class Voxelization(ProcessOp):
    def __init__(self, args: dict):
        super().__init__()
        self.resize = Resize(size=(args['resize_shape'], args['resize_shape']))

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
    def __init__(self, args: dict):
        super().__init__()
        # Knobs
        model_name = args['model_name']
        self.model = YOLO(f'{model_name}.pt')
        self.coco_category_to_id_dict = get_dict_coco_category_to_id()
        self.compute_latencies = [] 
        self.batch_accuraies = []
    
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data['images']))
        
        if profile_compute_latency:
            start_compute_time = time.time()
        
        # compute
        labels = batch_data['labels']
        org_image_shape = batch_data['org_image_shape']
        # batch_tensor = torch.tensor(np.array(batch_data['images']), dtype=torch.float32)
        batch_tensor = batch_data['images'] / 255.0
        results = self.model.predict(batch_tensor, save=False, verbose=False)
        
        if profile_compute_latency:
            self.compute_latencies.append(time.time()-start_compute_time)

        # Calculate Accuracy
        h, w = org_image_shape[0], org_image_shape[1]
        resize_matrix = torch.tensor([
            [w, 0, 0, 0],
            [0, h, 0, 0],
            [0, 0, w, 0],
            [0, 0, 0, h]
        ], dtype=torch.float32).to(device)
        mean_APs = []
        for result, label in zip(results, labels):
            raw_pred_clss = result.boxes.cls 
            raw_pred_confs = result.boxes.conf
            raw_pred_bboxes = result.boxes.xyxyn
            

            raw_pred_bboxes = torch.matmul(raw_pred_bboxes, resize_matrix).round().int().cpu()
            # print(result.boxes.xyxy)
            # exit(0)

            # filter unwanted boxes
            filtered_pred_bboxes, filtered_pred_clss, filtered_pred_confs = [], [], []
            pred_boxes = []
            for pred_cls, pred_conf, pred_bbox in zip(raw_pred_clss, raw_pred_confs, raw_pred_bboxes):
                pred_bbox = pred_bbox.tolist()
                pred_cls = int(pred_cls.item())
                if pred_cls not in self.coco_category_to_id_dict:
                    continue
                # filtered_pred_clss.append(self.coco_category_to_id_dict[pred_cls])
                # filtered_pred_confs.append(pred_conf)
                # filtered_pred_bboxes.append(pred_bbox)

                pred_boxes.append(pred_bbox + [self.coco_category_to_id_dict[pred_cls]] + [pred_conf])

            true_boxes = []
            for true_box in label:
                if true_box[4] == -1:
                    break
                else:
                    true_boxes.append(true_box)

            mean_AP = calculate_map(true_boxes=true_boxes, pred_boxes=pred_boxes, iou_threshold=0.5, num_classes=num_classes)

            mean_APs.append(mean_AP.item()) 

        self.batch_accuraies.append(sum(mean_APs)/len(mean_APs)) 
        return 

    def get_compute_latency(self):
        return sum(self.compute_latencies)/len(self.compute_latencies)

    def get_endpoint_accuracy(self):
        if len(self.batch_accuraies)  == 0:
            return 0
        return sum(self.batch_accuraies)/len(self.batch_accuraies)

    def calculate_batch_accuracy(self):
        pass