import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

a = [ 0,  0,  1,  2,  3,  0,  4,  0,  0,  2,  0,  5,  6,  7,  9, 10,  7, 10,  9, 11, 11, -1,  5,  5,
  6, 12, 12, 13,  6, 12, 14, 15,  6, 13, 13,  6, 14, 12, 15, 13, 13, 13, 17, 19, 17, 20, 21, 17,
 22, 18, 17, 17, 23, 17, 17, 23, 17, 17, 18, 17, 17, 17, 17, 24, 24, 22, 25, 24, 29, 31, 25, 31,
 29, 25, 32, 31, 29, 33, 24, 24, 34, 24, 29, 35, 25, 32, 29, 31, 25, 32, 36, 37, 38, 40, 40, 41,
 40, 43, 41, 36, 37, 39, 38, 41, 44, 45, 39, 36, 37, 39, 38, 36, 39, 50, 36, 39, 41, 40, 39, 37,
 38, 41, 36, 39, 38, 45, 44, 36, 45, 39, 38, 37, 36, 51, 39, 38, 37, 44, 41, 36, 39, 38, 37, 43,
 36, 51, 39, 36, 39, 36, 36, 43, 36, 36, 43, 39, 38, 39, 38, 37, 36, 43, 39, 38, 36, 43, 39, 36,
 25, 39, 38, 25, 32, 33, 24, 24, 34, 26, 35, 29, 53, 24, 53, 24, 33, 24, 24, 26, 35, 33, 24, 24,
 34, 26, 35, 33, 24, 24, 34, 24, 35, 33, 24, 24, 34, 35, 24, 24, 53, 24, 24, 24, 24, 24, 33, 24,
 24, 34, 26, 41, 41, 36, 37, 42, 45, 54, 55, 54, 54, 56, 54, 57,  7,  9, 54, 54, 55, 54, 54, 55,
 55, 58]

a = np.array(a)
print(np.unique(a), len(np.unique(a)))

exit(0)
pcd = o3d.io.read_point_cloud('pcds/lora_thresholding_0.5_after_cons.ply')

print(pcd)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.2, min_points=300, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
print(labels.shape)

colors = plt.get_cmap("gist_ncar")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.io.write_point_cloud('/home2/aneesh.chavan/Change_detection/pcds/dbscan_after_cons.ply', pcd)