from nuimages import NuImages
import tqdm
import cv2
import numpy as np
from ultralytics import YOLO 
import torch
from metric import calculate_map
from PIL import Image 
from utils import get_dict_nuimage_category_to_id
from numpy import ndarray


class Op:
    def __init__(self):
        pass

    def get_result(self):
        NotImplementedError()

class ProcessOp(Op):
    def profile(self, batch_data):
        """Do some transformation"""
        NotImplementedError()

class SourceOp(Op):
    def load_batch(self):
        """Do some transformation"""
        NotImplementedError() 

class Loader(SourceOp):
    def __init__(self, args: dict):
        self.root = '/nuimages/'
        
        self.nuim = NuImages(dataroot=self.root, version='v1.0-val', verbose=False, lazy=True)
        self.batch_size = args['batch_size']

        self.cur_batch_idx = 0
        self.nuim_category_to_id_dict = get_dict_nuimage_category_to_id()

    def load_batch(self):
        start_idx = self.cur_batch_idx * self.batch_size
        batch_imgs = []
        batch_labels = []
        for cur_idx in range(start_idx, start_idx + self.batch_size):
            sample = self.nuim.sample[cur_idx]
            img_fname = self.nuim.get('sample_data', sample['key_camera_token'])['filename']
            
            img_source = cv2.imread(self.root + img_fname)
            img_source = cv2.resize(img_source, (640, 640))
            img_source: ndarray = img_source.astype(np.float32) / 255.0

            
            batch_imgs.append(img_source)

            batch_labels.append(self.get_ground_truth_box(sample).tolist())
        
        # Transformation 
        batch_tensor = np.array(batch_imgs)
        batch_tensor = np.transpose(batch_tensor, (0, 3, 1, 2))

        return {
            'images': batch_tensor,
            'labels': batch_labels
        }

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


class Detector(ProcessOp):
    def __init__(self, args: dict):
        # Knobs
        model_name = args['model_name']
        self.model = YOLO(f'{model_name}.pt') 
    
    def profile(self, batch_data):
        # images = np.array(batch_data['images'])
        labels = batch_data['labels']
        batch_tensor = torch.tensor(np.array(batch_data['images']), dtype=torch.float32)
        
        results = self.model.predict(batch_tensor, save=False)
        return 
        print(images)

        return  

        # print(batch_data)
        for image, label in tqdm.tqdm(zip(images, labels)):
            results = self.model.predict(image, save=False)
            result = results[0]
            print(result)
        return  

    def get_recorded_data(self):
        return 
