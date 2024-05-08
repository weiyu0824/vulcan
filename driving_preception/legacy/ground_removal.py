# -*- coding: utf-8 -*-
# Filename : ground_removal
__author__ = 'Xumiao Zhang'

# Run remove ground from point cloud data

# Usage: python3 ground_removal.py -p [point cloud path] -s [save path]

import os
import numpy as np
import open3d as o3d
import argparse
from utils import *

def get_inliers(pcd_data, plane_model, point_err_thres=0.2):
    [a, b, c, d] = plane_model
    x = np.sum(pcd_data[:,:3] * np.array([a, b, c]), axis=1) + d
    inliers = np.argwhere(np.absolute(x) < point_err_thres).reshape(-1)
    return inliers

def get_ground_plane_ransac(pcd, distance_threshold, mask=None):
    pcd_o3d = pcd_np_to_o3d(pcd)
    plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    return plane_model, inliers, np.setdiff1d(np.arange(len(pcd)), inliers)

def get_ground_plane_naive(pcd, point_err_thres, ground_z=0):
    plane_model = [0, 0, 1, -ground_z]
    inliers = get_inliers(pcd, plane_model, point_err_thres=point_err_thres)
    return plane_model, inliers, np.setdiff1d(np.arange(len(pcd)), inliers)

def remove_ground(pcd, distance=0.1, alg='ransac'):
    if alg == 'ransac':
        _, indices_ground, indices_object  = get_ground_plane_ransac(pcd, distance)
    elif alg == 'naive':
        _, indices_ground, indices_object  = get_ground_plane_naive(pcd, distance)
    return pcd[indices_object], pcd[indices_ground]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pcd_path', type=str, default=None, help='Point cloud file')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='Path to save generated object-only point cloud')
    args = parser.parse_args()

    pcd_path = args.pcd_path
    save_path = args.save_path

    pcd = load_pcd(pcd_path, 5)
    print(np.shape(pcd))

    pcd_object, pcd_ground = remove_ground(pcd)
    print(np.shape(pcd_object), np.shape(pcd_ground))

    if save_path:
        filename = os.path.split(pcd_path)[-1].split('.')[0]
        save_pcd(pcd_object, os.path.join(save_path, f'{filename}_object.bin'))
        save_pcd(pcd_ground, os.path.join(save_path, f'{filename}_ground.bin'))
