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

from fpfh.utilities import PointCloudGen, get_angular_error


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

def register_point_clouds(source, target, voxel_size):
    source_down, source_fpfh = downsample_and_compute_fpfh(source, voxel_size)
    target_down, target_fpfh = downsample_and_compute_fpfh(target, voxel_size)

    distance_threshold = voxel_size * 1.5
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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))

    # Refine the registration using ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.2, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return result_icp.transformation

# class FPFH_RANSAC:
#     @staticmethod
#     def preprocess_point_cloud(pcd: PointCloudGen, voxel_size):
#         pcd_down = copy.deepcopy(pcd)

#         radius_normal = voxel_size * 2
#         pcd_down.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#         radius_feature = voxel_size * 5
#         pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             pcd_down,
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#         return pcd_down, pcd_fpfh

#     @staticmethod        
#     def run(source_pcd, target_pcd: PointCloudGen, voxel_size = 0.05):
#         source_down, source_fpfh = FPFH_RANSAC.preprocess_point_cloud(source_pcd, voxel_size)
#         target_down, target_fpfh = FPFH_RANSAC.preprocess_point_cloud(target_pcd, voxel_size)

#         distance_threshold = voxel_size * 2

#         result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#             source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#             4, [
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                     0.9),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                     distance_threshold)
#             ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 200))

#         estimated_transformation = result.transformation

#         return estimated_transformation