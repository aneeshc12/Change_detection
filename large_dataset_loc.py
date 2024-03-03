from object_memory import *
import ast, pickle, shutil, time
import psutil, os
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
    sampling_period: int = 80
    downsampling_rate: int = 5 # downsample points every these many frames
    save_dir: str = "/scratch/vineeth.bhat/vin-experiments/large_dataset_trials/8-rooms-new"
    start_file_index: int = 2
    last_file_index: int = -1
    rot_correction: float = 0.0 # keep as 30 for 8-room-new 
    look_around_range: int = 0 # number of sucessive frames to consider at every frame
    save_individual_objects: bool = False

    down_sample_voxel_size: float = 0.01 # best results
    create_ext_mesh: bool = False
    save_point_clouds: bool = False
    fpfh_global_dist_factor: float = 1.5
    fpfh_local_dist_factor: float = 0.4
    fpfh_voxel_size: float = 0.05
    localise_times: int = 1

    loc_results_start_file_index: int = 1
    loc_results_last_file_index: int = -1
    loc_results_sampling_period: int = 400

if __name__=="__main__":
    start_time = time.time()

    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    # creating save dir
    os.makedirs(largs.save_dir, exist_ok=True)
    print(f"Created save directory {largs.save_dir}")

    files = os.listdir(os.path.join(largs.test_folder_path, "rgb"))
    num_files = len(files)
    print(f"We have {num_files} files")

    print("\nBegin Memory Initialization")
    mem = ObjectMemory(device = largs.device, 
                       ram_pretrained_path=largs.ram_pretrained_path,
                       sam_checkpoint_path = largs.sam_checkpoint_path,
                       lora_path=largs.lora_path)
    print("Memory Init'ed\n")

    if largs.last_file_index == -1:
        largs.last_file_index = num_files
    if largs.loc_results_last_file_index == -1:
        largs.loc_results_last_file_index = num_files

    frame_counter = 0

    for cur_frame in range(largs.start_file_index, largs.last_file_index + 1, largs.sampling_period):
        for i in range(cur_frame, min(largs.last_file_index + 1, cur_frame + largs.look_around_range + 1)):
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

            mem.process_image(testname=f"view%d" % i, 
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

            print("\t ----------------")

            if frame_counter % largs.downsampling_rate == 0:
                if largs.down_sample_voxel_size > 0:
                    print(f"Downsampling at {frame_counter} frame voxel size as {largs.down_sample_voxel_size}")
                    mem.downsample_all_objects(voxel_size=largs.down_sample_voxel_size)

            frame_counter += 1

    if largs.down_sample_voxel_size > 0:
        print(f"Downsampling using voxel size as {largs.down_sample_voxel_size}")
        mem.downsample_all_objects(voxel_size=largs.down_sample_voxel_size)

    end_time = time.time()
    print(f"Traversal completed in {end_time - start_time} seconds")

    pcd_list = []
    
    for info in mem.memory:
        object_pcd = info.pcd
        pcd_list.append(object_pcd)

    combined_pcd = o3d.geometry.PointCloud()

    if largs.save_individual_objects:
        individual_mem_save_dir = os.path.join(largs.save_dir,
            f"ind_mems_{largs.start_file_index}_{largs.last_file_index}_{largs.sampling_period}_{largs.look_around_range}")
        os.makedirs(individual_mem_save_dir, exist_ok=True)

    for i in range(len(pcd_list)):
        pcd_np = pcd_list[i]
        pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = pcd_vec

        if largs.save_individual_objects:
            cur_save_path = os.path.join(individual_mem_save_dir, 
                f"memory_{i}.pcd") 

            o3d.io.write_point_cloud(cur_save_path, pcd)
            print(f"{i} pointcloud saved to", cur_save_path)

        combined_pcd += pcd

    save_path = os.path.join(largs.save_dir, 
        f"mem_{largs.start_file_index}_{largs.last_file_index}_{largs.sampling_period}_{largs.look_around_range}.pcd")
    o3d.io.write_point_cloud(save_path, combined_pcd)
    print("Memory's pointcloud saved to", save_path)

    print("\n\n\t---------------------")
    mem.view_memory()

    start_time = time.time()

    tgt = []
    pred = []
    trans_errors = []
    rot_errors = []
    chosen_assignments = []

    for i in range(largs.loc_results_start_file_index, 
                   largs.loc_results_last_file_index + 1, 
                   largs.loc_results_sampling_period):
        print(f"\n\tLocalizing image {i} currently")
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


        estimated_pose, chosen_assignment = mem.localise(image_path=image_file_path, 
                                        depth_image_path=depth_file_path,
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

        pred.append(estimated_pose.tolist())
        trans_errors.append(translation_error)
        rot_errors.append(rotation_error)
        chosen_assignments.append(chosen_assignment)

    for i in range(len(trans_errors)):
        print(f"Pose {i + 1}")
        print("Translation error", trans_errors[i])
        print("Rotation errors", rot_errors[i])

    end_time = time.time()
    print(f"Localization completed in {end_time - start_time} seconds")
