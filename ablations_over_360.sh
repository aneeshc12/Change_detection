#!/bin/bash --login
#SBATCH -n 16
#SBATCH -A research
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=10
#SBATCH --gres=gpu:1
#SBATCH --mail-user=aneesh.c@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -w gnode041

# module add cuda/8.0
# module add cudnn/7-cuda-8.0

# Define arrays for possible values
DOWNSAMPLE_VOXEL_SIZES=(0 0.01 0.001)
GLOB_DIST_VALUES=(1 1.5 2.0)
LOCAL_DIST_VALUES=(0.1 0.4 0.8 1.2)
FPFH_VOXEL_VALUES=(0.2)
USE_MESH=0

# conda init bash
conda activate reid
# /home2/$USER/Change_detection/download_weights.sh
mkdir -p /scratch/$USER/360_runs/

cd /home2/$USER/Change_detection

# Iterate over values
for i in 1 2 3 4; do
echo $i
done;

python runthrough_360.py

for DOWNSAMPLE_VOXEL_SIZE in "${DOWNSAMPLE_VOXEL_SIZES[@]}"; do
    for GLOB_DIST in "${GLOB_DIST_VALUES[@]}"; do
        for LOCAL_DIST in "${LOCAL_DIST_VALUES[@]}"; do
            # Continue if LOCAL_DIST is more than GLOB_DIST
            if (( $(awk 'BEGIN {print ("'"$LOCAL_DIST"'" > "'"$GLOB_DIST"'")}') )); then
                continue
            fi

            for FPFH_VOXEL in "${FPFH_VOXEL_VALUES[@]}"; do

                # Build ABLATION_NAME
                ABLATION_NAME="${DOWNSAMPLE_VOXEL_SIZE}_${USE_MESH}_${GLOB_DIST}_${LOCAL_DIST}_${FPFH_VOXEL}"

                echo $ABLATION_NAME

                # Your Python command here
                python runthrough_360.py \
                    --sam-checkpoint-path=/scratch/$USER/sam_vit_h_4b8939.pth \
                    --ram-pretrained-path=/scratch/$USER/ram_swin_large_14m.pth \
                    --test-folder-path=/home2/$USER/Change_detection/360_zip \
                    --memory-save-path=/scratch/$USER/360_runs/memory_${ABLATION_NAME}.pcd \
                    --down-sample-voxel-size=${DOWNSAMPLE_VOXEL_SIZE} \
                    --no-create-ext-mesh \
                    --save-results-path=/scratch/$USER/360_runs/results_${ABLATION_NAME}.json \
                    --fpfh-global-dist-factor=${GLOB_DIST} \
                    --fpfh-local-dist-factor=${LOCAL_DIST} \
                    --fpfh-voxel-size=${FPFH_VOXEL} 
                    # > /scratch/$USER/360_runs/run_${ABLATION_NAME}.txt

                echo "Test done!"
                exit 0

            done
        done
    done
done
