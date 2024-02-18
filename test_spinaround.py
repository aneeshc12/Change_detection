from object_memory import *
import numpy as np
from scipy.spatial.transform import Rotation as Rot

if __name__ == "__main__":
    tgt = []
    pred = []

    # get paths
    lora_path = 'models/vit_finegrained_5x40_procthor.pt'
    dataset_root = '/scratch/aneesh/procthor_spinaround/'
    
    # get poses
    with open('/scratch/aneesh/procthor_spinaround/pose/poses.txt', 'r') as f:
        p = json.load(f)
    p = np.array(p["corners"], dtype=np.float64)

    poses = np.zeros((8,7), dtype=np.float64)
    poses[:, :3] = p[:, :3]
    for i, zyx in enumerate(p):
        poses[i, 3:] = Rot.from_euler('zyx', zyx[3:], degrees=True).as_quat()

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
    mem = ObjectMemory(lora_path=lora_path)
    print("Memory Init'ed")

    for obj_num, obj in enumerate(objects):
        # treat each view as the query view
        for target in range(0,7):
            target_pose = None
            for i, pose in enumerate(poses):
                num = i+1

                print(f"Processing img %d" % num)
                t = view[:3]
                q = view[3:]
                if num == target:
                    target_pose = pose
                    continue
                
                mem.process_image(testname=f"view%d" % num, image_path=f"/scratch/aneesh/procthor_spinaround/rgb/%s/%d.png" % (obj, num), 
                                depth_image_path=f"/scratch/aneesh/procthor_spinaround/depth/%s/%d.npy" % (obj, num), pose=pose)
                print("Processed\n")

            mem.view_memory()

            estimated_pose = mem.localise(image_path=f"/scratch/aneesh/procthor_spinaround/rgb/%s/%d.png" % (obj, target), 
                                        depth_image_path=f"/scratch/aneesh/procthor_spinaround/depth/%s/%d.npy" % (obj, target))

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
        
        break
        

