import numpy as np
import copy
import open3d as o3d
from tqdm import tqdm
import os
import pickle

class PointCloudGen():

    def __init__(self, ply_path = None, 
                 init_rot = [0, 0, 0], 
                 init_trans = [0, 0, 0], 
                 add_noise = True,
                 noise_std_dev=0.01,
                 ):
        if ply_path is not None:
            self.pcd_original = o3d.io.read_point_cloud(ply_path)
            self.pcd_original.translate(-self.pcd_original.get_center())

            self.pcd_original.transform(
                self.__euler_angles_and_translation_to_transform_matrix(init_rot, init_trans)
            )

            self.pcd_original_diameter = np.linalg.norm(np.asarray(self.pcd_original.get_min_bound()) 
                                                        - np.asarray(self.pcd_original.get_max_bound()))
            
            if add_noise:
                self.pcd_original = self.__add_gaussian_noise_to_point_cloud(self.pcd_original, noise_std_dev)
            
            self.pcd_visible = None
        else:
            self.pcd_original = None
            self.pcd_visible = None
            self.pcd_original_diameter = 0

    def __add_gaussian_noise_to_point_cloud(self, pcd, noise_std_dev):
        """
        Add Gaussian noise to a point cloud.

        :param pcd: Open3D PointCloud object.
        :param noise_std_dev: Standard deviation of Gaussian noise.
        :return: Noisy PointCloud object.
        """
        # Generate Gaussian noise for each point
        noise = np.random.normal(0, noise_std_dev, size=(len(pcd.points), 3))
        
        # Add noise to the original points
        noisy_points = np.asarray(pcd.points) + noise

        # Create a new point cloud with the noisy points
        noisy_pcd = o3d.geometry.PointCloud()
        noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)
        
        if pcd.has_colors():
            noisy_pcd.colors = pcd.colors  # Copy colors from the original point cloud if available
        
        return noisy_pcd
    
    def __euler_angles_and_translation_to_transform_matrix(self, angles, translation):
        """
        Convert Euler angles (in radians) and translation to a 3D transformation matrix.

        :param angles: List or array of Euler angles [rx, ry, rz] for X, Y, and Z axes.
        :param translation: List or array representing translation [tx, ty, tz].
        :return: 4x4 transformation matrix.
        """
        rx, ry, rz = angles

        # Rotation matrix around X axis
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

        # Rotation matrix around Y axis
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

        # Rotation matrix around Z axis
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

        # Combine the rotation matrices in ZYX order
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        # Create a 4x4 transformation matrix with rotation and translation
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        return transform_matrix
    
    def downsample_original_pcd(self, voxel_size = 0.03):
        self.pcd_original = self.pcd_original.voxel_down_sample(voxel_size)

    def set_pcd_visible(self, ply_path, add_noise = False, noise_std_dev=0.01):
        self.pcd_visible = o3d.io.read_point_cloud(ply_path)
        if add_noise:
            self.pcd_visible = self.__add_gaussian_noise_to_point_cloud(self.pcd_visible, noise_std_dev)

    def gen_clouds_from_view(self, camera = None, radius = None, add_noise = True, noise_std_dev=0.01):

        if camera is None:
            camera = [0, self.pcd_original_diameter, 0]

        if radius is None:
            radius = self.pcd_original_diameter * 1000

        _, pt_map = self.pcd_original.hidden_point_removal(camera, radius)

        self.pcd_visible = self.pcd_original.select_by_index(pt_map)

        if add_noise:
            self.pcd_visible = self.__add_gaussian_noise_to_point_cloud(self.pcd_visible, noise_std_dev)

    def transform_visible_pc(self, rot_angles, trans):
        """
            Accepts in degrees
        """
        transformation = self.__euler_angles_and_translation_to_transform_matrix(
            np.radians(np.array(rot_angles)),
            trans
        )

        self.visible_pcd_transformation = transformation

        self.pcd_visible.transform(transformation)

    def view_all_pcds(self, view_axes = True, axes_size = 0.25, highlight_pcd_visible_colour = None):

        geometries_list = [self.pcd_original]

        if view_axes:
            mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0])
            geometries_list.append(mesh_coord_frame)

        if self.pcd_visible is not None:
            if highlight_pcd_visible_colour is not None:
                pcd_visible_copy = copy.deepcopy(self.pcd_visible)
                pcd_visible_copy.paint_uniform_color(highlight_pcd_visible_colour)
                geometries_list.append(pcd_visible_copy)
            else:
                geometries_list.append(self.pcd_visible)
            
        o3d.visualization.draw_geometries(geometries_list)

    def save_viewport_image(self, img_path, view_axes = True, axes_size = 0.25, highlight_pcd_visible_colour = None):
        geometries_list = [self.pcd_original]

        if view_axes:
            mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0])
            geometries_list.append(mesh_coord_frame)

        if self.pcd_visible is not None:
            if highlight_pcd_visible_colour is not None:
                pcd_visible_copy = copy.deepcopy(self.pcd_visible)
                pcd_visible_copy.paint_uniform_color(highlight_pcd_visible_colour)
                geometries_list.append(pcd_visible_copy)
            else:
                geometries_list.append(self.pcd_visible)

        vis = o3d.visualization.Visualizer()

        vis.create_window(visible=False)

        for pcd in geometries_list:
            vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        
        vis.capture_screen_image(img_path)
        vis.destroy_window()

    def load_from_dir(self, dir_path):
        pcd_original_path = os.path.join(dir_path, "full_pcd.pcd")
        pcd_visible_path = os.path.join(dir_path, "viewed_pcd.pcd")
        config_path = os.path.join(dir_path, "config.pkl")

        self.pcd_original = o3d.io.read_point_cloud(pcd_original_path)
        self.pcd_visible = o3d.io.read_point_cloud(pcd_visible_path)

        with open(config_path, "rb") as pickle_file:
            config_path = pickle.load(pickle_file)

        rotation = config_path["rotation"]
        translation = config_path["translation"]

        transformation = self.__euler_angles_and_translation_to_transform_matrix(
            np.radians(np.array(rotation)),
            translation
        )
        self.visible_pcd_transformation = transformation

    def expected_transformation(self):
        return np.linalg.inv(self.visible_pcd_transformation)

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T,    R_est)).trace() - 1) / 2, -1.0), 1.0)))