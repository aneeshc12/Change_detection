import os, sys, time
from object_memory import *
import ast, pickle, shutil
import psutil


@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    lora_path: str='models/vit_finegrained_5x40_procthor.pt'
    test_folder_path: str='/home2/aneesh.chavan/Change_detection/360_zip/'
    device: str='cuda'
    sam_checkpoint_path: str = '/scratch/aneesh.chavan/sam_vit_h_4b8939.pth'
    ram_pretrained_path: str = '/scratch/aneesh.chavan/ram_swin_large_14m.pth'
    memory_save_path: str = ''
    save_results_path: str = ''
    down_sample_voxel_size: float = 0.01 # best results
    create_ext_mesh: bool = False
    save_point_clouds: bool = False
    fpfh_global_dist_factor: float = 1.5
    fpfh_local_dist_factor: float = 0.4
    fpfh_voxel_size: float = 0.05
    localise_times: int = 1

    def to_dict(self):
        return {
            "lora_path": self.lora_path,
            "test_folder_path": self.test_folder_path,
            "device": self.device,
            "sam_checkpoint_path": self.sam_checkpoint_path,
            "ram_pretrained_path": self.ram_pretrained_path,
            "memory_save_path": self.memory_save_path,
            "save_results_path": self.save_results_path,
            "down_sample_voxel_size": self.down_sample_voxel_size,
            "create_ext_mesh": self.create_ext_mesh,
            "save_point_clouds": self.save_point_clouds,
            "fpfh_global_dist_factor": self.fpfh_global_dist_factor,
            "fpfh_local_dist_factor": self.fpfh_local_dist_factor,
            "fpfh_voxel_size": self.fpfh_voxel_size,
            "localise_times": self.localise_times
        }

if __name__ == "__main__":
    start_time = time.time()

    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    poses_json_path = os.path.join(largs.test_folder_path, "json_poses.json")

    tgt = []
    pred = []

    print("\nBegin Memory Initialization")
    mem = ObjectMemory(device = largs.device, 
                       ram_pretrained_path=largs.ram_pretrained_path,
                       sam_checkpoint_path = largs.sam_checkpoint_path,
                       lora_path=largs.lora_path)
    print("Memory Init'ed\n")

    with open(poses_json_path, 'r') as f:
        poses = json.load(f)

    for i, view in enumerate(poses["views"]):
        num = i+1

        print(f"Processing img %d" % num)
        q = Rotation.from_euler('zyx', [r for _, r in view["rotation"].items()], degrees=True).as_quat()
        t = np.array([x for _, x in view["position"].items()])
        pose = np.concatenate([t, q])
        
        mem.process_image(testname=f"view%d" % num, 
                            image_path = os.path.join(largs.test_folder_path, f"view%d/view%d.png" % (num, num)), 
                            depth_image_path=os.path.join(largs.test_folder_path,f"view%d/view%d.npy" % (num, num)), 
                            pose=pose)
        print("Processed\n")

    if largs.down_sample_voxel_size > 0:
        print(f"Downsampling using voxel size as {largs.down_sample_voxel_size}")
        mem.downsample_all_objects(voxel_size=largs.down_sample_voxel_size, use_external_mesh=largs.create_ext_mesh)

    # getting results
    tgt = []
    pred = []
    trans_errors = []
    rot_errors = []
    chosen_assignments = []
    for i, view in enumerate(poses["views"]):
        target_num = i+1

        print(f"Processing img %d" % target_num)
        q = Rotation.from_euler('zyx', [r for _, r in view["rotation"].items()], degrees=True).as_quat()
        t = np.array([x for _, x in view["position"].items()])
        target_pose = np.concatenate([t, q])
        tgt.append(target_pose)

        cur_estimated_poses = []
        cur_translation_errors = []
        cur_rotation_errors = []
        cur_chosen_assignments = []

        print(f"With {target_num} as target")
        for i in range(largs.localise_times):
            print(f"\tLocalize trial {i + 1} -----------------------")

            estimated_pose, chosen_assignment = mem.localise(image_path=os.path.join(largs.test_folder_path,f"view%d/view%d.png" % 
                                                                (target_num, target_num)), 
                                            depth_image_path=(os.path.join(largs.test_folder_path,"view%d/view%d.npy" % 
                                                                            (target_num, target_num))),
                                            save_point_clouds=largs.save_point_clouds,
                                            fpfh_global_dist_factor = largs.fpfh_global_dist_factor, 
                                            fpfh_local_dist_factor = largs.fpfh_global_dist_factor, 
                                            fpfh_voxel_size = largs.fpfh_voxel_size)

            print("Target pose: ", target_pose)
            print("Estimated pose: ", estimated_pose)

            translation_error = np.linalg.norm(target_pose[:3] - estimated_pose[:3]) 
            rotation_error = QuaternionOps.quaternion_error(target_pose[3:], estimated_pose[3:])

            print("Translation error: ", translation_error)
            print("Rotation_error: ", rotation_error)

            cur_estimated_poses.append(estimated_pose.tolist())
            cur_translation_errors.append(translation_error)
            cur_rotation_errors.append(rotation_error)
            cur_chosen_assignments.append(chosen_assignment)

            print("----\n")

        pred.append(cur_estimated_poses)
        trans_errors.append(cur_translation_errors)
        rot_errors.append(cur_rotation_errors)
        chosen_assignments.append(cur_chosen_assignments)

    end_time = time.time()
    print(f"360zip test completed in {(end_time - start_time)//60} minutes, {(end_time - start_time)%60} seconds")

    # Print the total GPU memory, allocated memory, and free memory
    cuda_memory_stats = torch.cuda.memory_stats()
    max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
    print(f"Max GPU memory usage: {max_cuda_memory_GBs:.3f} GB")

    # Print the memory usage in bytes, kilobytes, and megabytes
    pid = psutil.Process()
    memory_info = pid.memory_info()
    memory_info_GBs = memory_info.rss / (1e3 ** 3)
    print(f"Memory usage: {memory_info_GBs:.3f} GB")

    # saving memory to scratch
    if largs.memory_save_path != "":
        pcd_list = []
        for info in mem.memory:
            object_pcd = info.pcd
            pcd_list.append(object_pcd)

        combined_pcd = o3d.geometry.PointCloud()

        for pcd_np in pcd_list:
            pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
            pcd = o3d.geometry.PointCloud()
            pcd.points = pcd_vec
            combined_pcd += pcd

        os.makedirs(os.path.dirname(largs.memory_save_path), exist_ok=True)
        o3d.io.write_point_cloud(largs.memory_save_path, combined_pcd)
        print("Pointcloud saved to", largs.memory_save_path)

    trans_rmses = []
    rot_rmses = []

    print("\n\nFinal results:")
    for i in range(len(trans_errors)):
        print(f"Pose {i + 1}")
        print("Translation error", trans_errors[i])
        print("Rotation errors", rot_errors[i])

        cur_trans_rmse = np.sqrt(np.mean(np.array(trans_errors[i])**2))
        cur_rot_rmse = np.sqrt(np.mean(np.array(rot_errors[i])**2))

        print("Translation rmse", cur_trans_rmse)
        print("Rotatiion rmse", cur_rot_rmse)

        trans_rmses.append(cur_trans_rmse)
        rot_rmses.append(cur_rot_rmse)


    # saving other results
    if largs.save_results_path != "":
        os.makedirs(os.path.dirname(largs.save_results_path), exist_ok=True)

        results = {
            "peak_gpu_usage": max_cuda_memory_GBs,
            "memory_usage": memory_info_GBs,
            "total_time": end_time - start_time,
            "chosen_assignments": chosen_assignments,
            "target_poses": [arr.tolist() for arr in tgt],
            "estimated_poses": pred,
            "translation_error": trans_errors,
            "rotation_error": rot_errors,
            "translation_rmses": trans_rmses,
            "rotation_rmses": rot_rmses,
            "largs": largs.to_dict(),
        }

        with open(largs.save_results_path, 'w') as json_file:
            json.dump(results, json_file)

        print(f"Saved results to {largs.save_results_path}")

    torch.cuda.empty_cache()