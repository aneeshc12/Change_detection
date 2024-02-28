# Define arrays for possible values
DOWNSAMPLE_VOXEL_SIZES=(0 0.01 0.001)
GLOB_DIST_VALUES=(1 1.5 2.0)
LOCAL_DIST_VALUES=(0.1 0.4 0.8 1.2)
FPFH_VOXEL_VALUES=(0.02 0.05 0.1)
USE_MESH=0
LOC_TIMES=10

# Iterate over values
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
                    --test-folder-path=/home2/$USER/Documents/aneesh_thread/Change_detection/360_zip \
                    --memory-save-path=/scratch/$USER/vin-experiments/360_runs/memory_${ABLATION_NAME}.pcd \
                    --down-sample-voxel-size=${DOWNSAMPLE_VOXEL_SIZE} \
                    --no-create-ext-mesh \
                    --save-results-path=/scratch/$USER/vin-experiments/360_runs/results_${ABLATION_NAME}.json \
                    --fpfh-global-dist-factor=${GLOB_DIST} \
                    --fpfh-local-dist-factor=${LOCAL_DIST} \
                    --fpfh-voxel-size=${FPFH_VOXEL} \
                    --localise-times=${LOC_TIMES} \
                    > /scratch/$USER/vin-experiments/360_runs/run_${ABLATION_NAME}.txt

            done
        done
    done
done