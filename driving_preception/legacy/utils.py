# -*- coding: utf-8 -*-
# Filename : utils
__author__ = 'Xumiao Zhang'

# Utility functions

import numpy as np
import open3d as o3d

def pcd_np_to_o3d(pcd_np):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:,:3])
    pcd_o3d.paint_uniform_color([0, 0, 1])
    return pcd_o3d

def pcd_o3d_to_np(pcd_o3d):
    pcd_np = np.array(cloud_o3d.points)
    pcd_np = np.hstack((pcd_np, np.array(cloud_o3d.colors)))
    return pcd_np

def load_pcd(pcd_path, ndim=5, float_type=np.float32):
    pcd = np.fromfile(pcd_path, dtype=float_type, count=-1)
    # print(pcd)
    pcd = pcd.reshape([-1,ndim])
    return pcd



def save_pcd(pcd, save_path):
    with open(save_path, 'w') as f:
        pcd.tofile(f)
