from object_memory import *
import ast, pickle, shutil, time
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

    memory_sampling_period: int = 40
    memory_save_dir: str = "/scratch/vineeth.bhat/vin-experiments/large_dataset_trials/8-rooms-new"
    memory_start_file_index: int = 1
    memory_last_file_index: int = -1

    rot_correction: float = 0.0 # keep as 30 for 8-room-new 
    look_around_range: int = 1 # number of sucessive frames to consider at every frame
    
    loc_sampling_period: int = 40
    loc_save_dir: str = "/scratch/vineeth.bhat/vin-experiments/large_dataset_trials/8-rooms-new"
    loc_start_file_index: int = 1
    loc_last_file_index: int = -1

if __name__=="__main__":
    raise




