import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def transform_pcd_to_global_frame(pcd, pose):
    t = pose[:3]
    q = pose[3:]

    q /= np.linalg.norm(q)                  # normalise
    R = Rotation.from_quat(q).as_matrix()


    transformed_pcd = R @ pcd
    transformed_pcd += t.reshape(3, 1)

    return transformed_pcd

full = o3d.geometry.PointCloud()


btxt = [
{'position': {'x': -2.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 180.00001525878906, 'z': 0.0}},
{'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 135.00001525878906, 'z': 0.0}},
{'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 3.25}, 'rotation': {'x': -0.0, 'y': 90.00000762939453, 'z': 0.0}},
{'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': -0.0, 'y': 45.000003814697266, 'z': 0.0}},
{'position': {'x': -2.75, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}},
{'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}},
{'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 3.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}},
{'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': 0.0, 'y': 224.99998474121094, 'z': 0.0}},]

# btxt = [
# {'position': {'x': -2.5, 'y': 0, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 180.00001525878906, 'z': 0.0}},
# {'position': {'x': -4.5, 'y': 0, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 135.00001525878906, 'z': 0.0}},
# {'position': {'x': -4.5, 'y': 0, 'z': 3.25}, 'rotation': {'x': -0.0, 'y': 90.00000762939453, 'z': 0.0}},
# {'position': {'x': -4.5, 'y': 0, 'z': 0}, 'rotation': {'x': -0.0, 'y': 45.000003814697266, 'z': 0.0}},
# {'position': {'x': -2.75, 'y': 0, 'z': 0}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}},
# {'position': {'x': -0.5, 'y': 0, 'z': 0}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}},
# {'position': {'x': -0.5, 'y': 0, 'z': 3.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}},
# {'position': {'x': -0.5, 'y': 0, 'z': 6.25}, 'rotation': {'x': 0.0, 'y': 224.99998474121094, 'z': 0.0}},]


def conv(p):
    pose = np.empty(7)
    pos = p['position']
    rot = p['rotation']
    pose[:3] = [pos['x'], pos['y'], pos['z']]

    q = Rotation.from_euler('zyx', [rot['x'], rot['y'], rot['z']], degrees=True).as_quat()
    q /= np.linalg.norm(q)
    pose[3:] = q
    return pose

allPoses = [conv(p) for p in btxt]
for p in enumerate(allPoses):
    print(p[0], " | ", p[1][:3], "\n\t", p[1][3:])

# load pcds
def getDepth(depth_image_path, f=300):
    if depth_image_path == None:
        raise NotImplementedError
    else:
        depth_image = np.load(depth_image_path)
        
        w, h = depth_image.shape

        depth_image[depth_image > 18] = 0

        horizontal_distance = np.tile(np.linspace(-h/2, h/2, h, dtype=np.float32), (w,1))
        vertical_distance =   np.tile(np.linspace(-w/2, w/2, w, dtype=np.float32).reshape(1,-1).T, (1, h))

        X = horizontal_distance * depth_image/f
        Y = vertical_distance * depth_image/f
        Z = -depth_image

        # plt.imshow(depth_image)
        # plt.show()

        # plt.imshow(X)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(Y)
        # plt.show()

        # plt.imshow(Z)
        # plt.show()

        # combine caluclated X,Y,Z points
        all_pointclouds = np.stack([X, Y, Z]).reshape(3, -1)
        # all_pointclouds = all_pointclouds[:, all_pointclouds[1, :] < 1.2]
        # all_pointclouds = all_pointclouds[:, all_pointclouds[1, :] > -0.7]

        return all_pointclouds

pcds = []
for a in range(1,9):
    filename = f"360_zip/view%d/view%d.npy" % (a, a)
    k = o3d.geometry.PointCloud()
    pts = getDepth(filename)
    k.points = o3d.utility.Vector3dVector(pts.T)
    pcds.append(k)

# unregistered
# o3d.visualization.draw_geometries(pcds)


# resgister
# pcds[1].translate((2,0,0))
num = 10
for pose,pcd in zip(allPoses, pcds[:num]):
    bigR = pcd.get_rotation_matrix_from_quaternion(pose[3:])
    pcd.rotate(bigR.T, center=(0,0,0))
    pcd.translate(-pose[:3])


# bigR = pcds[0].get_rotation_matrix_from_xyz((0, np.pi, 0))
# pcds[0].rotate(bigR.T, center=(0,0,0))

# bigR = pcds[1].get_rotation_matrix_from_xyz((0, np.pi * 135/180., 0))
# pcds[1].rotate(bigR.T, center=(0,0,0))

# pcds[0].translate((-2.5, 0.9, 6.25))
# pcds[1].translate((-4.5, 0.9, 6.25))


mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
o3d.visualization.draw_geometries([mesh] + pcds[:num])