from object_memory import *
import ast, pickle, shutil
import psutil

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
    sampling_period: int = 40
    save_dir: str = "/scratch/vineeth.bhat/large_dataset_trials/8-rooms-new"
    start_file_index: int = 1
    last_file_index: int = -1
    rot_correction: float = 0.0 # keep as 30 for 8-room-new 
    look_around_range: int = 3 # number of sucessive frames to consider at every frame

if __name__=="__main__":
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

    pcd_list = []
    
    for obj_id, info in mem.memory.items():
        object_pcd = info.pcd
        pcd_list.append(object_pcd)

    combined_pcd = o3d.geometry.PointCloud()

    for pcd_np in pcd_list:
        pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = pcd_vec
        combined_pcd += pcd

    save_path = f"{largs.save_dir}/mem_{largs.start_file_index}_{largs.last_file_index}_{largs.sampling_period}_{largs.look_around_range}.pcd"
    o3d.io.write_point_cloud(save_path, combined_pcd)
    print("Memory's pointcloud saved to", save_path)


