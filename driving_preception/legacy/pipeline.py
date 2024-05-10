import os
import time
import numpy as np
import argparse
import tqdm
import pickle
import json

from utils import load_pcd, save_pcd
from ground_removal import remove_ground
from voxelization import voxelize_o3d
import open3d as o3d
# from object_detection import Inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--ground_para', type=float, default=0.1, help='Ground removal parameter: distance threshold')
    parser.add_argument('-v', '--voxel_para', type=float, default=0.1, help='Voxelization parameter: distance threshold')
    parser.add_argument("--save", action='store_true', default=False, help='save output point cloud')
    args = parser.parse_args()

    distance = args.ground_para
    voxel_size = args.voxel_para

    # pcd_path_path = "/home/xumiao/mmdetection3d/data/nuscenes_vulcan_val_lidarpath.txt"
    # pcd_paths = np.loadtxt(pcd_path_path, dtype=str)
    base_path = '/dev/shm/wylin2/'
    input_base_path = os.path.join(base_path, 'standard')
    output_base_path = os.path.join(base_path, f'g{str(int(distance*100))}-v{str(int(voxel_size*100))}')
    os.makedirs(output_base_path, exist_ok=True)
    
    # Define paths for subfolders
    # maps_path = os.path.join(output_base_path, "maps")
    point_cloud_folders = ['samples', 'sweeps']
    radar_folders = ['LIDAR_TOP']

    for folder_name in point_cloud_folders:
        # Create sub-subfolders inside the folder
        folder_path = os.path.join(output_base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for rader_folder_name in radar_folders:
            sub_sub_folder_path = os.path.join(output_base_path, folder_name, rader_folder_name)
            print(sub_sub_folder_path)
            os.makedirs(sub_sub_folder_path, exist_ok=True) 
    # exit()

    for folder_name in point_cloud_folders:

        for rader_folder_name in radar_folders:
            # Get the full path of the current radar folder
            input_folder_path = os.path.join(input_base_path, folder_name, rader_folder_name)
            output_folder_path = os.path.join(output_base_path, folder_name, rader_folder_name) 
            files = os.listdir(input_folder_path)
            
            # print(files)

            gnd_input_size = []
            vox_input_size = []
            det_input_size = []
            gnd_compute_latencies = []
            vox_compute_latencies = []

            for pcd_file in tqdm.tqdm(files):
                pcd_input_path = input_folder_path + '/' + pcd_file
                pcd_output_path = output_folder_path + '/' + pcd_file 

                # load pcd
                pcd = load_pcd(pcd_input_path, 5)
                # pcd = o3d.io.read_point_cloud(pcd_input_path)
                # o3d.visualization.draw_geometries([pcd])

                # print(np.shape(pcd))
            
                # ground removal
                gnd_input_size.append(len(pickle.dumps(pcd)))
                t1 = time.time()
                pcd_object, pcd_ground = remove_ground(pcd, distance)
                t2 = time.time()
                gnd_compute_latencies.append(t2-t1)

                # voxelization
                vox_input_size.append(len(pickle.dumps(pcd_object)))
                t3 = time.time()
                pcd_voxelized = voxelize_o3d(pcd_object, voxel_size)
                t4 = time.time()
                vox_compute_latencies.append(t4-t3)

                # detection
                det_input_size.append(np.shape(pcd_voxelized)[0])

                pcd = pcd_voxelized

                # save_pcd(pcd, pcd_output_path)
                # print('save')
                # break
            
            with open('profile_result.json', 'r') as fp:
                records = json.load(fp)
            with open('profile_result.json', 'w') as fp:
                records.append(dict(
                    ground_factor=args.ground_para,
                    voxel_factor =args.voxel_para,
                    result={
                        'ground_removal_compute_latency': np.mean(gnd_compute_latencies),
                        'voxelization_compute_latency': np.mean(vox_compute_latencies),
                        'ground_removal_input_size': np.mean(gnd_input_size),
                        'voxelization_input_size': np.mean(vox_input_size),
                        'detector_input_size': np.mean(det_input_size)
                    } 
                ))
                json.dump(records, fp)
            exit(0)