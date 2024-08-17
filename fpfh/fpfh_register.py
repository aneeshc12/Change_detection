import os
import numpy as np
import pandas as pd
import copy
import open3d as o3d
import cv2
import pickle
from tqdm import tqdm
from PIL import Image
import shutil
import time

from scipy.spatial.transform import Rotation as R

def downsample_and_compute_fpfh(pcd, voxel_size):
    # Downsample the point cloud using Voxel Grid
    pcd_down = copy.deepcopy(pcd)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def register_point_clouds(source, target, voxel_size, global_dist_factor = 1.5, local_dist_factor = 0.4):
    try:    # catch cases where normals cant be computed
        source_down, source_fpfh = downsample_and_compute_fpfh(source, voxel_size)
        target_down, target_fpfh = downsample_and_compute_fpfh(target, voxel_size)

        distance_threshold = voxel_size * global_dist_factor
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.99))

        # Refine the registration using ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size*local_dist_factor, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
    except:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size*local_dist_factor, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    return result_icp.transformation, result_icp.inlier_rmse, result_icp.fitness

def evaluate_transform(source, target, trans_init, threshold=0.02):
    res = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )

    return res.inlier_rmse, res.fitness


if __name__ == '__main__':
    # read and split pcd
    pcdA = o3d.io.read_point_cloud('bunny.ply')

    ptsA = np.asarray(pcdA.points)

    A_left = ptsA[:len(ptsA)//2, :]
    A_right = ptsA[len(ptsA)//2:, :]

    A_right += [0.1,0,0]

    labels = np.zeros(len(ptsA))
    # labels[len(ptsA)//2:] = 1 
    print(labels)
    print(A_left.shape, A_right.shape, ptsA.shape)

    # target pcd
    pcdA_left = o3d.geometry.PointCloud()
    pcdA_right = o3d.geometry.PointCloud()

    pcdA_left.points = o3d.utility.Vector3dVector(A_left)
    pcdA_right.points = o3d.utility.Vector3dVector(A_right)

    pcdA_left.paint_uniform_color([1.,0.,0.])
    pcdA_right.paint_uniform_color([1.,1.,0.])

    pcdA_split = pcdA_left + pcdA_right


    # source pcd
    pcdB_left = o3d.geometry.PointCloud()
    pcdB_right = o3d.geometry.PointCloud()

    pcdB_left.points = o3d.utility.Vector3dVector(A_left)
    pcdB_right.points = o3d.utility.Vector3dVector(A_right)

    pcdB_left.paint_uniform_color([0.,1.,0.])
    pcdB_right.paint_uniform_color([0.,0.,1.])

    pcdB_split = pcdB_left + pcdB_right

    # transform
    randomT = np.eye(4)
    randomT[:3,:3] = R.from_euler('xyz', [0,20,0], degrees=True).as_matrix() 
    randomT[:3, 3] = [2.,0.,0.]

    from copy import deepcopy as dc
    pcdB_split_og = dc(pcdB_split)
    pcdB_split.transform(randomT)

    print(randomT)

    a = np.asarray(pcdA_split.points)
    b = np.asarray(pcdB_split.points)

    T, rmse, fitness = register_point_clouds(pcdA_split, pcdB_split, voxel_size=0.05)
    T_sem, rmse_sem, fitness_sem = register_point_clouds_with_semantics(pcdA_split, pcdB_split, labels, labels, voxel_size=0.05)

    # T_semantic, _ = semantic_icp(a, b, labels, labels, tolerance=0.00001)
    # T_base_icp, _, _ = icp(a, b, tolerance=0.00001)

    o3d.io.write_point_cloud('tx_gt.ply', pcdA_split + pcdB_split)

    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(T[:3, :3])
    T_inv[3, :3] = -T[3, :3]
    
    o3d.io.write_point_cloud('tx_aligned.ply', pcdA_split + pcdB_split.transform(T_inv))
    
    print("------------------------------------")
    print(randomT)
    print()
    print(T)
    print()
    print(T_sem)
    # print("semantic: ", T_semantic)
    # print("base: ", T_base_icp)
    print(f"Semantic is: {np.isclose(randomT, T_sem).all()}, base icp is {np.isclose(randomT, T).all()}")
    # print(f"Normal ICP is: {np.isclose(randomT, T).all()}")
