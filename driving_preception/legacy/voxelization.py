# -*- coding: utf-8 -*-
# Filename : voxelization
__author__ = 'Xumiao Zhang'

# Run voxelization to sample point cloud data

# Usage: python3 voxelization.py -p [point cloud path] -s [save path]

import os
import numpy as np
import open3d as o3d
import argparse
from utils import *

def voxelize_o3d(pcd, voxel_size=0.05):
    pcd_o3d = pcd_np_to_o3d(pcd)
    pcd_o3d_voxelized, _, indices = pcd_o3d.voxel_down_sample_and_trace(voxel_size, pcd_o3d.get_min_bound(), pcd_o3d.get_max_bound(), False)
    indices = [int_vec.pop() for int_vec in indices]
    return pcd[indices]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pcd_path', type=str, default=None, help='Point cloud file')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.1, help='Voxel size')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='Path to save generated object-only point cloud')
    args = parser.parse_args()

    pcd_path = args.pcd_path
    voxel_size = args.voxel_size
    save_path = args.save_path

    pcd = load_pcd(pcd_path, 5)
    print(np.shape(pcd))

    pcd_voxelized = voxelize_o3d(pcd, voxel_size)
    print(np.shape(pcd_voxelized))

    if save_path:
        filename = os.path.split(pcd_path)[-1].split('.')[0]
        save_pcd(pcd_voxelized, os.path.join(save_path, f'{filename}_voxelized.bin'))
