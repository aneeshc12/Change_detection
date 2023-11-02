# Change detection repo, contains the codebase used to obtain results so far

## Run sequence
1. `python data_assoc.py`
2. `python choose_images.py`
3. `python data.py` to create a train/val/test split for the images
4. `python lora.py` for training LoRA ViT and showing results