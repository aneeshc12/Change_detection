import open3d as o3d
import numpy as np
import os
from glob import glob

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return outlier_cloud + inlier_cloud



pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()

for file in [
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view2_chair.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view2_sofa.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view2_table.npy'
]:
    points = np.load(file)
    print(file, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(np.random.random(3))

    # statistical outlier detection
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
    cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.05)
    pcd = display_inlier_outlier(pcd, ind)
    pcd1 += pcd

for file in [
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view3_chair.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view3_sofa.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/2pcds/view3_table.npy'
]:
    points = np.load(file)
    print(file, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(np.random.random(3))

    # statistical outlier detection
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
    cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.05)
    pcd = display_inlier_outlier(pcd, ind)
    pcd2 += pcd

# pcd1.paint_uniform_color([1, 0.706, 0])
# pcd2.paint_uniform_color([0, 0.706, 1])

o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd2])
o3d.visualization.draw_geometries([pcd1, pcd2])
