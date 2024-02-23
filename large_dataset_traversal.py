from object_memory import *
import ast, pickle

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
    sampling_frequency: int = 4000
    save_dir: str = "/scratch/vineeth.bhat/large_dataset_trials/8-rooms"
    start_file_index: int = 1

if __name__=="__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    files = os.listdir(os.path.join(largs.test_folder_path, "rgb"))
    num_files = len(files)
    print(f"We have {num_files} files")

    print("\nBegin Memory Initialization")
    mem = ObjectMemory(device = largs.device, 
                       ram_pretrained_path=largs.ram_pretrained_path,
                       sam_checkpoint_path = largs.sam_checkpoint_path,
                       lora_path=largs.lora_path)
    print("Memory Init'ed\n")

    for i in range(largs.start_file_index, num_files+1, largs.sampling_frequency):
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
                "x": pose_dict[1]['x'] + 30,
                "y": pose_dict[1]['y'],
                "z": pose_dict[1]['z']
            }
        }

        print(f"Processing img %d" % i)
        q = Rotation.from_euler('xyz', [r for _, r in pose_dict["rotation"].items()], degrees=True).as_quat()
        t = np.array([x for _, x in pose_dict["position"].items()])
        pose = np.concatenate([t, q])

        mem.process_image(testname=f"view%d" % i, 
                            image_path = image_file_path, 
                            depth_image_path = depth_file_path, 
                            pose=pose)
        print("Processed\n")

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

    save_path = f"{largs.save_dir}/output_{largs.sampling_frequency}.pcd"
    o3d.io.write_point_cloud(save_path, combined_pcd)
    print("Pointcloud saved to", save_path)


