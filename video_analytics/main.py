from op import Loader, Detector, NuImageDataset, Voxelization
import socket
import json
import os
import tqdm
import time
from torch.utils.data import DataLoader
import argparse
from pipeline import Pipeline, BatchData, MeanAveragePrecisionEvaluator
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

def get_pipeline_args(): 
    return {
        "model": os.environ.get('model'),
        "resize_factor": os.environ.get('resize_factor')
    }

def load_cache(pipe_args):
    dump_filename = '_'.join([str(i) for i in pipe_args.values()])
    with open(f"./cache/{dump_filename}.json", 'r') as f:
        dump = json.load(f)
    return dump

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

    for index, data in enumerate(dataloader):
        batch_data = {
            'fname': data[2],
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
    
def profile_pipeline_normal():
    detector_args = get_detector_args()
    voxelization_args = get_voxelization_args()
    voxelization = Voxelization((900, 1600), voxelization_args)
    detector = Detector((900, 1600), detector_args)
    
    profile_result = {}

    org_image_shape = (900, 1600)
    nuimage_data = NuImageDataset()

    # pipeline = Pipeline("video_analytics", [voxelization, detector], MeanAveragePrecisionEvaluator(), None)

    print('start profile this pipeline ...', voxelization_args, detector_args)
    start_time = time.time()

    print('profile accuracy')
    # Profile args:
    batch_size = 64
    num_workers = 8
    dataloader = DataLoader(nuimage_data, batch_size=1, shuffle=False, num_workers=num_workers)

    eval_result = {}
    for data in dataloader:
        batch_data = {
            'images': data[0], 
            'labels': data[1],
            'org_image_shape': org_image_shape 
        }
        batch_data = voxelization.profile(batch_data)
        batch_data = detector.profile(batch_data) 
        fname = data[2][0]
        last_accuracy = detector.get_last_accuracy()
        # print(f'Accuracy: {last_accuracy} for {fname}')
        eval_result.update({fname: last_accuracy})
        
    dump = {
        "args" : {
            "voxelization": voxelization_args,
            "detector": detector_args
        },
        "results" : eval_result
    }
    model = os.environ.get('model')
    resize = os.environ.get('resize_factor')
    with open(f"./cache/{model}_{resize}.json", "w") as fp:
        json.dump(dump, fp)
    return profile_result    

def profile_pipeline_cached():
    cache = load_cache(get_pipeline_args())
    detector_args = get_detector_args()
    voxelization_args = get_voxelization_args()
    voxelization = Voxelization((900, 1600), voxelization_args)
    detector = Detector((900, 1600), detector_args)
    
    pipeline = Pipeline("video_analytics", [voxelization, detector], MeanAveragePrecisionEvaluator(), cache)
    
    profile_result = {}

    org_image_shape = (900, 1600)
    nuimage_data = NuImageDataset()

    print('profile ', get_pipeline_args())
    # Profile args:
    batch_size = 1
    num_workers = 8
    dataloader = DataLoader(nuimage_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    acc = []
    cul_acc = []  
    for data in dataloader:
        fname = data[2][0]
        result = pipeline.run_cached(fname)
        acc.append(result)
        cul_acc.append(sum(acc) / len(acc))

    profile_result['args'] = get_pipeline_args()
    profile_result['accuracy'] = acc
    profile_result['cummulative_accuracy'] = cul_acc

    return profile_result
    
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
        num = 1
        for i in range(num):
            for rf in resize_factors:
                for model in models:
                    os.environ["resize_factor"] = str(rf)
                    os.environ["model"] = str(model)
                    result = profile_pipeline_cached() 
                    records.append(dict(
                        rf=rf,
                        model=model,
                        result=result
                    ))
        
        with open ('./result/profile_result.json', 'w') as fp:
            json.dump(records, fp)


