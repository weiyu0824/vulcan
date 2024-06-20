from op import Loader, Detector, NuImageDataset, Voxelization
import socket
import json
import os
import tqdm
import time
from torch.utils.data import DataLoader
import argparse
# import torch

def get_detector_args():
    return {
        'model_name': os.environ.get('model', 'yolov8x'),
        'resize_shape': int(os.environ.get('resize_shape', 640))
    }
def get_voxelization_args():
    return {
       'resize_factor': float(os.environ.get('resize_factor', 1))
    }
def get_loader_args():
    return {
        'batch_size': 1,
    }

def profile_pipeline():
    detector_args = get_detector_args()
    voxelization_args = get_voxelization_args()
    voxelization = Voxelization((900, 1600), voxelization_args)
    detector = Detector((900, 1600), detector_args)
    
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
    profile_result['100_sample_accuracy'] = detector.get_endpoint_accuracy()
    
    print(profile_result)
    # return profile_result
    # exit(0) 


    print('profile accuracy')
    # Profile args:
    batch_size = 64
    num_workers = 8
    dataloader = DataLoader(nuimage_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    cum_accuracy = []  
    for data in tqdm.tqdm(dataloader):
        batch_data = {
            'images': data[0], 
            'labels': data[1],
            'org_image_shape': org_image_shape 
        }
        batch_data = voxelization.profile(batch_data)
        batch_data = detector.profile(batch_data) 

        cum_accuracy.append(detector.get_endpoint_accuracy())

   
    profile_result['accuracy'] = detector.get_endpoint_accuracy()
    profile_result['profile_latency'] = time.time() - start_time
    profile_result['cummulative_accuracy'] = cum_accuracy 
    profile_result['batch_accuracy'] = detector.get_batch_accuracy()

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
    
    
    parser = argparse.ArgumentParser(description='Project management CLI')
    parser.add_argument('--method', '-m', choices=['build', 'profile'], required=True, help='Method to execute')


    args = parser.parse_args()
    if args.method == 'build':
        loader = Loader()
        loader.build_index()
    elif args.method == 'profile':
        records = []
        resize_factors = [0.8]
        models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

        for rf in resize_factors:
            for model in models:
                os.environ["resize_factor"] = str(rf)
                os.environ["model"] = str(model)
                result = profile_pipeline() 
                print(result)
                records.append(dict(
                    rf=rf,
                    model=model,
                    result=result
                ))
        
        with open ('profile_result.json', 'w') as fp:
            json.dump(records, fp)


