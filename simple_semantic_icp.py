import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        print("distances : ", distances.shape)
        print("indices : ", indices.shape)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

"""
A: Nx3
B: Mx3
"""

def correspondences_given_labels(A, B, labels_A, labels_B):
    with torch.no_grad():
        high_value = 1e8

        unique_labels_A = np.unique(labels_A)
        unique_labels_B = np.unique(labels_B)

        dist_matrix = torch.cdist(A, B, p=2.0)  # NxM
        mask = torch.tensor(labels_A[:, np.newaxis] != labels_B[np.newaxis, :]).bool()

        # print(dist_matrix.shape, mask.shape)
        dist_matrix.masked_fill_(mask, high_value)


        # correspondences = torch.argmin(dist_matrix, axis=-1)
        distance, correspondences = torch.min(dist_matrix, -1)
        # print(dist_matrix)      # AxB
        # print(distance)
        return correspondences, distance


def semantic_icp(A, B, labels_A, labels_B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: (Nxm) numpy array of source mD points
        B: (Nxm) numpy array of destination mD point
        labels_A: (N) numpy array of source labels
        labels_B: (N) numpy array of destination labels
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape[0] == labels_A.shape[0]
    assert B.shape[0] == labels_B.shape[0]
    # assert len(np.unique(labels_A)) == len(np.unique(labels_B))

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # print("asdfsadfsdf: ", src.shape, dst.shape)
        indices, distances = correspondences_given_labels(torch.tensor(src[:m,:].T), torch.tensor(dst[:m,:].T), labels_A, labels_B)
        
        # print("indices : ", indices.shape)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        # print(distances.shape, distances)
        mean_error = np.mean(np.array(distances))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(np.asarray(A), src[:m,:].T)

    return T

import open3d as o3d
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    # read and split pcd
    pcdA = o3d.io.read_point_cloud('bunny.ply')

    ptsA = np.asarray(pcdA.points)

    A_left = ptsA[:len(ptsA)//2, :]
    A_right = ptsA[len(ptsA)//2:, :]

    A_right += [0.1,0,0]

    labels = np.zeros(len(ptsA))
    labels[len(ptsA)//2:] = 1 
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

    # o3d.io.write_point_cloud('tx_split.ply', pcdA_left + pcdA_right)

    k = correspondences_given_labels(torch.tensor(np.asarray(pcdA_split.points)), torch.tensor(np.asarray(pcdB_split.points)), torch.tensor(labels), torch.tensor(labels))
    print("adsf: " ,k)

    # transform
    randomT = np.eye(4)
    randomT[:3,:3] = R.from_euler('xyz', [0,0,0], degrees=True).as_matrix() 
    randomT[:3, 3] = [2.,0.,0.]

    from copy import deepcopy as dc
    pcdB_split_og = dc(pcdB_split)
    pcdB_split.transform(randomT)

    print(randomT)

    a = np.asarray(pcdA_split.points)
    b = np.asarray(pcdB_split.points)

    T_semantic, _ = semantic_icp(a, b, labels, labels, tolerance=0.00001)
    T_base_icp, _, _ = icp(a, b, tolerance=0.00001)


    o3d.io.write_point_cloud('tx_gt.ply', pcdA_split + pcdB_split)

    T = T_semantic

    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(T[:3, :3])
    T_inv[3, :3] = -T[3, :3]
    
    o3d.io.write_point_cloud('tx_aligned.ply', pcdA_split + pcdB_split.transform(T_inv))
    
    print("------------------------------------")
    print(randomT)
    print()
    print("semantic: ", T_semantic)
    print("base: ", T_base_icp)
    print(f"Semantic is: {np.isclose(randomT, T_semantic).all()}, base icp is {np.isclose(randomT, T_base_icp).all()}")
