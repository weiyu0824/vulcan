from op import Loader, Detector, NuImageDataset, Voxelization
import socket
import json
import os
import tqdm
import time
from torch.utils.data import DataLoader
import torch

def get_detector_args():
    return {
        'model_name': os.environ.get('model', 'yolov8x')
    }
def get_voxelization_args():
    return {
       'resize_shape': int(os.environ.get('resize_shape', '640')) 
    }
def get_loader_args():
    return {
        'batch_size': 1,
        'resize_shape': int(os.environ.get('resize_shape', '640'))
    }

def my_collate(batch):
    # print(type(batch[0][0]))
    # print(batch[0][0])
    # exit()
    # data = torch.stack(batch[:, 0])
    data = [item for item in batch]
    # target = [item[1] for item in batch]
    # data, target = [], []
    # for img  in batch:
    #     data.append(img)
        # target.append(box)
    # target = torch.LongTensor(target)
    return data

def profile_pipeline():
    detector_args = get_detector_args()
    voxelization_args = get_voxelization_args()
    voxelization = Voxelization(voxelization_args)
    detector = Detector(detector_args)
    
    profile_result = {}

    org_image_shape = (900, 1600)
    nuimage_data = NuImageDataset()


    print('start profile this pipeline ...', voxelization_args, detector_args)
    start_time = time.time()

    print('profile latency & input size')
    # Profile args:
    num_profile_batch = 100 #Cab't be small because of warm-up
    dataloader = DataLoader(nuimage_data, batch_size=1, shuffle=False) 

    for index, data in tqdm.tqdm(enumerate(dataloader)):
        batch_data = {
            'images': data[0], 
            'labels': data[1],
            'org_image_shape': org_image_shape 
        }
        batch_data = voxelization.profile(batch_data, profile_input_size=True, profile_compute_latency=True)
        batch_data = detector.profile(batch_data, profile_input_size=True, profile_compute_latency=True)

        if index >= num_profile_batch:
            break

    profile_result['voxelization_input_size'] = voxelization.get_input_size()
    profile_result['voxelization_compute_latency'] = voxelization.get_compute_latency()
    profile_result['detector_input_size'] = detector.get_input_size()
    profile_result['detector_compute_latency'] = detector.get_compute_latency()
    print(profile_result)
    


    print('profile accuracy')
    # Profile args:
    batch_size = 64
    num_workers = 8
    dataloader = DataLoader(nuimage_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    for data in tqdm.tqdm(dataloader):
        batch_data = {
            'images': data[0], 
            'labels': data[1],
            'org_image_shape': org_image_shape 
        }
        batch_data = voxelization.profile(batch_data)
        batch_data = detector.profile(batch_data) 


   
    profile_result['accuracy'] = detector.get_endpoint_accuracy()
    profile_result['profile_latency'] = time.time() - start_time
    

    return profile_result
    
def start_connect(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        s.settimeout(5) 
        print('start connecnting...')

        s.connect((host, port))  # Example TCP port
        
        # Receive start profiling command
        cmd = s.recv(1024)
        print(cmd)
        
        print('start profiling ... ')

        profile_results = profile_pipeline()

        print('end profiling')

        
        s.send(json.dumps(profile_results).encode())


if __name__ == "__main__":
    # addr, port = 'localhost', 12343
    # start_connect(addr, port)

    result = profile_pipeline() 
    print(result)

    # version = 'v1.0-val'
    # with open(f'{version}-idx.json', 'r') as fp:
    #     index_data = json.load(fp)
    # max_num_box = 0
    # for data in tqdm.tqdm(index_data):
    #     box = data["true_boxes"]
    #     if max_num_box < len(box):
    #         max_num_box = len(box)
    # print(max_num_box)