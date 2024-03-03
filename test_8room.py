from object_memory import *
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import re, os

def getAllPoses(root="/scratch/aneesh.chavan/8room/8-room-v1/1/pose", maxn=3316):
    poses = []
    for i in tqdm(range(1,maxn)):
        try:
            with open(f"{root}/{str(i)}.txt") as f:
                k = re.sub("'","\"",f.read())
                j = json.loads(k)
                poses.append([j[0]['x'], j[0]['y'], j[0]['z'], j[1]['x'], j[1]['y'], j[1]['z']])
        except:
            continue
    poses = np.array(poses)
    return poses

if __name__ == "__main__":
    tgt = []
    pred = []
    
    # get paths
    lora_path = 'models/vit_finegrained_5x40_procthor.pt'
    dataset_root = '/scratch/aneesh.chavan/8room/8-room-v1'
    arrangement = "1"
    
    # get poses
    json_poses = getAllPoses(os.path.join(dataset_root, arrangement, "pose"))
    poses = np.zeros((json_poses.shape[0],7), dtype=np.float64)
    for i, pose in enumerate(json_poses):
        poses[i, :3] = pose[:3]
        poses[i, 3:] = Rot.from_euler('xyz', pose[3:], degrees=True).as_quat()

    # filter poses by a factor of 10, split these into training and test
    start_idx = 0
    end_idx = 100
    filtered_indices = [i for i in range(start_idx, end_idx, 2)]

    test_indices = filtered_indices[::5]
    train_indices = [i for i in filtered_indices if i not in test_indices]

    print(test_indices[:10])
    print(train_indices[:20])
    print("Train and test indices: ", len(train_indices), len(test_indices))

    # init memory
    print("Begin")
    mem = ObjectMemory(device='cuda', sam_checkpoint_path='/scratch/aneesh.chavan/sam_vit_h_4b8939.pth', ram_pretrained_path='/scratch/aneesh.chavan/ram_swin_large_14m.pth', lora_path=lora_path)
    print("Memory Init'ed")

    # assemble memory from train_indices
    for num in train_indices:
        print(f"Processing img %d" % num)
        
        pose = poses[num]
        print(pose)

        t = pose[:3]
        q = pose[3:]
        
        mem.process_image(testname=f"%s_view{num}" % ("8room-v1"), image_path=f"/scratch/aneesh.chavan/8room/8-room-v1/{arrangement}/rgb/{num}.png", 
                        depth_image_path=f"/scratch/aneesh.chavan/8room/8-room-v1/{arrangement}/depth/{num}.npy", pose=pose,
                        verbose=False, add_noise=False)
        print("Processed\n")
        mem.view_memory()


    # test each index in test_indices
    for i, target in enumerate(test_indices[:4]):
        print(f"Localising image {target}")
        target_pose = poses[target]
        estimated_pose = mem.localise(testname=str("8room-v1") ,image_path=f"/scratch/aneesh.chavan/8room/8-room-v1/{arrangement}/rgb/{target}.png", 
                                    depth_image_path=f"/scratch/aneesh.chavan/8room/8-room-v1/{arrangement}/depth/{target}.npy",
                                    save_point_clouds=True)
        tgt.append(target_pose)
        pred.append(estimated_pose)

    mem.clear_memory()

    # for _, m in mem.memory.items():
    #     np.save(f"pcds/new%d.npy" % m.id, m.pcd)
    # torch.cuda.empty_cache()

    for i, t, p in zip(test_indices, tgt, pred):
        print("Test index: ", i)
        print("Target pose:", t)
        print("Estimated pose:", p)
        print()

        
        

