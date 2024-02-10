import open3d as o3d
import numpy as np
import os
from glob import glob


pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()

for file in [
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view2_chair.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view2_sofa.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view2_table.npy'
]:
    points = np.load(file)
    print(file, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(np.random.random(3))
    pcd1 += pcd

for file in [
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view3_chair.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view3_sofa.npy',
    '/media/aneesh/Ubuntu_storage/RRC/Change_detection/pcds/view3_table.npy'
]:
    points = np.load(file)
    print(file, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(np.random.random(3))
    pcd2 += pcd

# pcd1.paint_uniform_color([1, 0.706, 0])
# pcd2.paint_uniform_color([0, 0.706, 1])

o3d.visualization.draw_geometries([pcd1, pcd2])
