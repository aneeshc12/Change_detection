from object_memory import *
import numpy as np
from scipy.spatial.transform import Rotation as Rot

if __name__ == "__main__":
    tgt = []
    pred = []
    
    # get paths
    lora_path = 'models/vit_finegrained_5x40_procthor.pt'
    dataset_root = '/scratch/aneesh.chavan/procthor_spinaround/'
    
    # get poses
    p = np.load(os.path.join(dataset_root, "pose.npy"))

    poses = np.zeros((8,7), dtype=np.float64)
    for i, pose in enumerate(p):
        poses[i, :3] = pose[:3]
        poses[i, 3:] = Rot.from_euler('xyz', pose[3:], degrees=True).as_quat()

    # list objects
    objects = [
        "armchairs",
        "chairs",
        "coffee_tables",
        "dining_tables",
        "floor_lamps",
        "garbage_cans",
        "side_tables",
        "sofas",
        "tv_stands"
    ]

    print("Begin")
    mem = ObjectMemory(device='cuda', sam_checkpoint_path='/scratch/aneesh.chavan/sam_vit_h_4b8939.pth', ram_pretrained_path='/scratch/aneesh.chavan/ram_swin_large_14m.pth', lora_path=lora_path)
    print("Memory Init'ed")

    for obj_num, obj in enumerate(objects):
        print(obj, " beign processed")

        # treat each view as the query view
        for target in range(0,8):
            target_pose = None


            for i, pose in enumerate(poses):
                num = i
                print(f"Processing img %d" % num)
                print(pose)
                if num == target:
                    target_pose = pose
                    continue

                t = pose[:3]
                q = pose[3:]
                
                mem.process_image(testname=f"%s_view%d" % (obj, num), image_path=f"/scratch/aneesh.chavan/procthor_spinaround/rgb/%s/%d.png" % (obj, num), 
                                depth_image_path=f"/scratch/aneesh.chavan/procthor_spinaround/depth/%s/%d.npy" % (obj, num), pose=pose)
                print("Processed\n")

            mem.view_memory()

            estimated_pose = mem.localise(testname=str(obj) ,image_path=f"/scratch/aneesh.chavan/procthor_spinaround/rgb/%s/%d.png" % (obj, target), 
                                        depth_image_path=f"/scratch/aneesh.chavan/procthor_spinaround/depth/%s/%d.npy" % (obj, target))

            print("Target pose: ", target_pose)
            print("Estimated pose: ", estimated_pose)

            mem.clear_memory()

            tgt.append(target_pose)
            pred.append(estimated_pose)

            # for _, m in mem.memory.items():
            #     np.save(f"pcds/new%d.npy" % m.id, m.pcd)
            torch.cuda.empty_cache()

        print("Object: ", obj)
        for i, t, p in zip(range(0,8), tgt, pred):
            print("Pose: ", i)
            print("Target pose:", t)
            print("Estimated pose:", p)
            print()

        tgt = []
        pred = []
        
        

