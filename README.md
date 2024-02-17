# Change detection repo, contains the codebase used to obtain results so far

## Setting up the environment

1. `conda create -n FinderV2 python=3.10`
2. `conda activate FinderV2`
3. `pip install -r requirements.txt`

## Run sequence

1. `python data_assoc.py`
2. `python choose_images.py`
3. `python data.py` to create a train/val/test split for the images
4. `python lora.py` for training LoRA ViT and showing results
