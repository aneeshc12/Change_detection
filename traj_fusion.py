import os, sys, time
from object_memory import *
import ast, pickle, shutil
import psutil, gc
  

@dataclass
class LocalArgs:
    """

    Class to hold local configuration arguments.

    """

    lora_path: str='models/vit_finegrained_5x40_procthor.pt'
    test_folder_path: str='/scratch/vineeth.bhat/8-room-new'
    device: str='cuda'
    sam_checkpoint_path: str = '/scratch/vineeth.bhat/sam_vit_h_4b8939.pth'
    ram_pretrained_path: str = '/scratch/vineeth.bhat/ram_swin_large_14m.pth'
    down_sample_voxel_size: float = 0.01 # best results
    create_ext_mesh: bool = False
    fpfh_global_dist_factor: float = 1.5
    fpfh_local_dist_factor: float = 0.4
    fpfh_voxel_size: float = 0.05
    rot_correction: float = 0.0 # keep as 30 for 8-room-new
    look_around_range: int = 0 # keep as 0 only
    downsampling_rate: int = 5 # downsample points every these many frames
    # parameters for creating memories
    source_start_file_index: int = 2
    source_last_file_index: int = 1100
    source_sampling_period: int = 80
    target_start_file_index: int = 822
    target_last_file_index: int = 1700
    target_sampling_period: int = 80
    save_dir: str = "/scratch/vineeth.bhat/vin-experiments/traj-fusion/8-rooms-new"
    inits_to_consider: int = 5 # how many initializations to consider


if __name__ == "__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    # creating save dir
    os.makedirs(largs.save_dir, exist_ok=True)
    print(f"Created save directory {largs.save_dir}")

    files = os.listdir(os.path.join(largs.test_folder_path, "rgb"))
    num_files = len(files)
    print(f"We have {num_files} files")

    start_time = time.time()    

    print("\nBegin Memory Initialization")
    source_mem = ObjectMemory(device = largs.device,
    ram_pretrained_path=largs.ram_pretrained_path,
    sam_checkpoint_path = largs.sam_checkpoint_path,
    lora_path=largs.lora_path)
    print("Source Memory Init'ed\n")

    frame_counter = 0

    for cur_frame in range(largs.source_start_file_index, largs.source_last_file_index + 1, largs.source_sampling_period):
        for i in range(cur_frame, min(largs.source_last_file_index + 1, cur_frame + largs.look_around_range + 1)):

            print(f"\n\tSeeing image {i} currently")
            image_file_path = os.path.join(largs.test_folder_path,
            f"rgb/{i}.png")
            depth_file_path = os.path.join(largs.test_folder_path,
            f"depth/{i}.npy")
            pose_file_path = os.path.join(largs.test_folder_path,
            f"pose/{i}.txt")

            

            with open(pose_file_path, 'r') as file:
                pose_dict = file.read()

            pose_dict = ast.literal_eval(pose_dict)
            pose_dict = {
                "position": {
                "x": pose_dict[0]['x'],
                "y": pose_dict[0]['y'],
                "z": pose_dict[0]['z']
                },
                "rotation": {
                "x": pose_dict[1]['x'] + largs.rot_correction,
                "y": pose_dict[1]['y'],
                "z": pose_dict[1]['z']
                }
            }

            q = Rotation.from_euler('xyz', [r for _, r in pose_dict["rotation"].items()], degrees=True).as_quat()
            t = np.array([x for _, x in pose_dict["position"].items()])
            pose = np.concatenate([t, q])

            
            source_mem.process_image(testname=f"view%d" % i,
            image_path = image_file_path,
            depth_image_path = depth_file_path,
            pose=pose)

            pid = psutil.Process()
            memory_info = pid.memory_info()
            memory_info_GBs = memory_info.rss / (1e3 ** 3)
            print(f"Memory usage: {memory_info_GBs:.3f} GB")

            

            cuda_memory_stats = torch.cuda.memory_stats()
            max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
            print(f"Max GPU memory usage: {max_cuda_memory_GBs:.3f} GB")

            if frame_counter % largs.downsampling_rate == 0:
                if largs.down_sample_voxel_size > 0:
                    print(f"Downsampling at {frame_counter} frame voxel size as {largs.down_sample_voxel_size}")

                    source_mem.downsample_all_objects(voxel_size=largs.down_sample_voxel_size)

            frame_counter += 1

    

    if largs.down_sample_voxel_size > 0:
        print(f"Downsampling using voxel size as {largs.down_sample_voxel_size}")
        source_mem.downsample_all_objects(voxel_size=largs.down_sample_voxel_size)

    end_time = time.time()
    print(f"Source traversal completed in {end_time - start_time} seconds")

    

    source_pcd_list = []

    for info in source_mem.memory:
        object_pcd = info.pcd
        source_pcd_list.append(object_pcd)

    source_combined_pcd = o3d.geometry.PointCloud()

    for i in range(len(source_pcd_list)):
        pcd_np = source_pcd_list[i]
        pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = pcd_vec
        source_combined_pcd += pcd

    

    source_mem_save_path = os.path.join(largs.save_dir,
    f"source_mem_{largs.source_start_file_index}_{largs.source_last_file_index}_{largs.source_sampling_period}_{largs.look_around_range}.pcd")

    o3d.io.write_point_cloud(source_mem_save_path, source_combined_pcd)
    print("Source memory's pointcloud saved to", source_mem_save_path)

    torch.cuda.empty_cache()
    print("Cleared cuda cache")

    ######### target

    """
    For every every image in the target
    trajectory, we process it and
    localise it to the memory

    then use fitnesses to get best initializations
    (account for number of points)
    and then run global registration
    """  


    gc.collect()
    pid = psutil.Process()
    memory_info = pid.memory_info()
    memory_info_GBs = memory_info.rss / (1e3 ** 3)
    print(f"Memory usage: {memory_info_GBs:.3f} GB")

    cuda_memory_stats = torch.cuda.memory_stats()
    max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
    print(f"Max GPU memory usage: {max_cuda_memory_GBs:.3f} GB")


    print("\n\n\t---------------------")
    print("In source trajectory, we have")
    source_mem.view_memory()

    start_time = time.time()

    tgt = []
    pred = []
    trans_errors = []
    rot_errors = []
    chosen_assignments = []
    fits = []
    frame_num_list = []
    adjusted_fits = []

    for i in range(largs.target_start_file_index,
    largs.target_last_file_index + 1,
    largs.target_sampling_period):
        print(f"\n\tLocalizing target image {i} currently")
        frame_num_list.append(i)
        image_file_path = os.path.join(largs.test_folder_path,
        f"rgb/{i}.png")
        depth_file_path = os.path.join(largs.test_folder_path,
        f"depth/{i}.npy")
        pose_file_path = os.path.join(largs.test_folder_path,
        f"pose/{i}.txt")

        with open(pose_file_path, 'r') as file:
            pose_dict = file.read()

        pose_dict = ast.literal_eval(pose_dict)
        pose_dict = {
            "position": {
            "x": pose_dict[0]['x'],
            "y": pose_dict[0]['y'],
            "z": pose_dict[0]['z']
            },
            "rotation": {
            "x": pose_dict[1]['x'] + largs.rot_correction,
            "y": pose_dict[1]['y'],
            "z": pose_dict[1]['z']
            }
        }

        q = Rotation.from_euler('xyz', [r for _, r in pose_dict["rotation"].items()], degrees=True).as_quat()
        t = np.array([x for _, x in pose_dict["position"].items()])
        target_pose = np.concatenate([t, q])
        tgt.append(target_pose)

        ret = source_mem.localise(image_path=image_file_path,
        depth_image_path=depth_file_path,
        save_point_clouds=False,
        fpfh_global_dist_factor = largs.fpfh_global_dist_factor,
        fpfh_local_dist_factor = largs.fpfh_global_dist_factor,
        fpfh_voxel_size = largs.fpfh_voxel_size)

        print(ret)
        raise

        if estimated_pose is None:
            print("No objects found in image; skipping")
            pred.append(np.array([1e10, 1e10, 1e10, 0, 0, 0, 1]))
            trans_errors.append(1e10)
            rot_errors.append(1e10)
            chosen_assignments.append([0])
            fits.append(-1e10)
            adjusted_fits.append(-1e10)
            continue

        print("Target pose: ", target_pose)
        print("Estimated pose: ", estimated_pose)
        print("Registration fit: ", fit)
        # adj_fit = fit * (len(chosen_assignment) ** 0.25)
        print("Adjusted fit: ", adj_fit)

        translation_error = np.linalg.norm(target_pose[:3] - estimated_pose[:3])
        rotation_error = QuaternionOps.quaternion_error(target_pose[3:], estimated_pose[3:])

        print("Translation error: ", translation_error)
        print("Rotation_error: ", rotation_error)

        
        pred.append(estimated_pose.tolist())
        trans_errors.append(translation_error)
        rot_errors.append(rotation_error)
        chosen_assignments.append(chosen_assignment)
        fits.append(fit)
        adjusted_fits.append(adj_fit)

        gc.collect()
        pid = psutil.Process()
        memory_info = pid.memory_info()
        memory_info_GBs = memory_info.rss / (1e3 ** 3)
        print(f"Memory usage: {memory_info_GBs:.3f} GB")

        cuda_memory_stats = torch.cuda.memory_stats()
        max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
        print(f"Max GPU memory usage: {max_cuda_memory_GBs:.3f} GB")

    

    for i in range(len(trans_errors)):
        print(f"Pose {i + 1}")
        print("Translation error", trans_errors[i])
        print("Rotation errors", rot_errors[i])
        print("Registration fit", fits[i])
        print("Adjusted fit", adjusted_fits[i])

    end_time = time.time()
    print(f"Fusion fits computed in {end_time - start_time} seconds")   
    

    gc.collect()
    pid = psutil.Process()
    memory_info = pid.memory_info()
    memory_info_GBs = memory_info.rss / (1e3 ** 3)
    print(f"Memory usage: {memory_info_GBs:.3f} GB")    

    cuda_memory_stats = torch.cuda.memory_stats()
    max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
    print(f"Max GPU memory usage: {max_cuda_memory_GBs:.3f} GB")    

    print("\n\n\n\t")
    print("Checking which transformations have lowest fits:")    

    def top_k_indices(lst, k = 3):
        enumerated_list = list(enumerate(lst))

        sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=True)

        top_k_indices = [index for index, value in sorted_list[:k]]

        return top_k_indices

    best_fit_indices = top_k_indices(adjusted_fits, largs.inits_to_consider)

    for i in best_fit_indices:

        print("\n--------------------\n")

        

        print(f"Index {i}")

        print(f"Frame {frame_num_list[i]}")

        print(f"fit {fits[i]}")

        print("Adjusted fit", adjusted_fits[i])

        

        print("\nTranslation error", trans_errors[i])

        print("Rotation errors", rot_errors[i])

        print("Registration fit", fits[i])

        print(f"Estimated transform {pred[i]}")