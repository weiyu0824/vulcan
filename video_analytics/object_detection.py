from op import Op

from nuimages import NuImages
import tqdm
import cv2
import numpy as np
from ultralytics import YOLO 
import torch
from metric import calculate_map



class ObjectDetector(Op):
    def __init__(self, args: dict):
        model_name = args['model_name']
        self.model = YOLO(f'{model_name}.pt') 
    
    def profile(self, batch_data):
        
        results = self.model.predict(root + filename, save=False)
        result = results[0]



        return  









def draw(img_path, bboxes, save_path):
    img = cv2.imread(img_path)
    color = (0, 255, 0)  # BGR color format (green)
    thickness = 2 
    for bbox in bboxes:
        # print('cool', bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness) 
    cv2.imwrite(save_path, img)

# class matcher
num_classes = 2
NuimageCategoryToId = {}
CocoCategoryToId = {}
person_id = 0
vehicle_id = 1
NuimageCategoryToId['human.pedestrian.adult'] = person_id
NuimageCategoryToId['human.pedestrian.child'] = person_id
NuimageCategoryToId['human.pedestrian.construction_worker'] = person_id
NuimageCategoryToId['human.pedestrian.personal_mobility'] = person_id
NuimageCategoryToId['human.pedestrian.police_officer'] = person_id
NuimageCategoryToId['human.pedestrian.stroller'] = person_id
NuimageCategoryToId['human.pedestrian.wheelchair'] = person_id
NuimageCategoryToId['vehicle.bicycle'] = vehicle_id
NuimageCategoryToId['vehicle.car'] = vehicle_id
NuimageCategoryToId['vehicle.motorcycle'] = vehicle_id 
NuimageCategoryToId['vehicle.truck'] = vehicle_id 
NuimageCategoryToId['vehicle.bus.rigid'] = vehicle_id
NuimageCategoryToId['vehicle.bus.bendy'] = vehicle_id
NuimageCategoryToId['vehicle.trailer'] = vehicle_id
CocoCategoryToId[0] = person_id
CocoCategoryToId[1] = vehicle_id
CocoCategoryToId[2] = vehicle_id
CocoCategoryToId[3] = vehicle_id
CocoCategoryToId[5] = vehicle_id
CocoCategoryToId[7] = vehicle_id


# Load data
root = 'nuscenes-devkit/data/sets/nuimages/'
nuim = NuImages(dataroot=root, version='v1.0-mini', verbose=False, lazy=True)
num_frame = len(nuim.sample)





def measure(model_name, iou_theshold):


    # Load model
    # model_name =
    #  'yolov8m'
    model = YOLO(f'{model_name}.pt')
    model.info()

    def get_ground_truth_box(sample):
        if not sample:
            raise ValueError('sample should not be none')
        key_camera_token = sample['key_camera_token']
        # object_tokens, surface_tokens = nuim.list_anns(sample['token'])
        bboxes = [o['bbox'] for o in nuim.object_ann if o['sample_data_token'] == key_camera_token]
        categories = [nuim.get('category', o['category_token'])['name']
                    for o in nuim.object_ann if o['sample_data_token'] == key_camera_token]
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
            if category not in NuimageCategoryToId:
                # print('filter: ', category)
                continue
            filtered_categories.append(NuimageCategoryToId[category])
            filtered_bboxes.append(bbox)
            # true_boxes.append(torch.cat([torch.tensor(bbox), torch.tensor(NuimageCategoryToId[category])]))
        # print(true_boxes)
        
        filtered_bboxes = torch.tensor(filtered_bboxes)
        filtered_categories = torch.unsqueeze(torch.tensor(filtered_categories), dim=1) 
        true_boxes = torch.cat([filtered_bboxes, filtered_categories], dim=1) 
        # print('true', true_boxes)
        # exit(0)
        return true_boxes 


    # metric
    pre_latency, inf_latency, post_latency = [], [], []
    mean_APs = []
    for i in range(num_frame):
        sample = nuim.sample[i]
        # key_camera_token = sample['key_camera_token']

        # # Get object annos & category
        # bboxes, categories = [], []
        # for obj_ann in nuim.object_ann:
        #     if obj_ann['sample_data_token'] == 'key_camera_token':
        #         bboxes.append(obj_ann['bbox'])
        #         categories.append(nuim.get('category', obj_ann['category_token']))

        # Get filename
        filename = nuim.get('sample_data', sample['key_camera_token'])['filename']

        # inference
        # import time 
        # st = time.time()
        # print(st)
        results = model.predict(root + filename, save=False)
        result = results[0]
        # pred_bboxes = results[0].boxes.xyxyn.numpy()   
        # print(results)
        # print(time.time() - st)

        # calculate latency 
        pre_latency.append(result.speed['preprocess'])
        inf_latency.append(result.speed['inference'])
        post_latency.append(result.speed['postprocess'])

        raw_pred_clss = result.boxes.cls 
        raw_pred_confs = result.boxes.conf
        raw_pred_bboxes = result.boxes.xyxy.round().int()

        # print(pred_confs)
        filtered_pred_bboxes, filtered_pred_clss, filtered_pred_confs = [], [], []
        for pred_cls, pred_conf, pred_bbox in zip(raw_pred_clss, raw_pred_confs, raw_pred_bboxes):
            
            pred_cls = int(pred_cls.item())
            # print(pred_cls)
            # print(CocoCategoryToId)
            if pred_cls not in CocoCategoryToId:
                continue
            filtered_pred_clss.append(CocoCategoryToId[pred_cls])
            filtered_pred_confs.append(pred_conf)
            filtered_pred_bboxes.append(pred_bbox)
        
        # print(filtered_pred_bboxes)
        # print(filtered_pred_clss)
        # print(filtered_pred_confs)
        if len(filtered_pred_bboxes) == 0:
            pred_boxes = []
        else:
            # print(filtered_pred_confs)
            # print(filtered_pred_clss)
            attr = torch.stack([torch.tensor(filtered_pred_clss), torch.tensor(filtered_pred_confs)]).T
            # print(attr)
            # print(filtered_pred_bboxes)
            pred_boxes = torch.cat([torch.stack(filtered_pred_bboxes), attr], dim=1 )


        # gt boxes=
        true_boxes = get_ground_truth_box(sample)
        # print(true_boxes)
        # print(pred_boxes)
        # exit(0)
        # print(true_boxes)
        # print('cur frame:', i, 'num gt', len(true_boxes), 'num de', len(pred_boxes))
        mean_AP = calculate_map(true_boxes=true_boxes, pred_boxes=pred_boxes, iou_threshold=0.5, num_classes=num_classes)
        # print('mAP: ', mean_AP.item())    
        mean_APs.append(mean_AP.item()) 
        # import pdb
        # breakpoint()

        # break
        # print(mean_AP)
        # exit()
        # calculate mAP
        # calculate_map(true_boxes=, pred_boxes=)
    print(mean_APs)
    return sum(mean_APs)/num_frame, sum(pre_latency)/num_frame, sum(inf_latency)/num_frame, sum(post_latency)/num_frame


def main():
    print(result)
    
if __name__ == "__main__":
    main()