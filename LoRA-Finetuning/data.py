import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dir = '/scratch/sarthak.chittawar/LoRA_Dataset/'

save_dir = '/scratch/sarthak.chittawar/LoRA'

if not os.path.exists(dir):
    print("No source directory. Quitting.")

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    
    c = 1
    print('Moving cropped files into single folder...')

    for i in ['ArmChairs', 'Chairs', 'Tables']:
        os.mkdir(save_dir+'/'+i)
        for j in os.listdir(dir+i):
            path = dir+i+'/'+j+'/cropped'
            for k in os.listdir(path):
                if not os.path.isdir(save_dir+'/'+i+'/'+k):
                    os.mkdir(save_dir+'/'+i+'/'+k)
                for z in os.listdir(path+'/'+k):
                    img = Image.open(path+'/'+k+'/'+z)
                    img.save('/scratch/sarthak.chittawar/LoRA/{}/{}/{}.png'.format(i, k, c))
                    c += 1
                    
else:
    print("File structure already exists. Proceeding with Train/Val/Test split.")
    os.mkdir(save_dir+'/train')
    os.mkdir(save_dir+'/val')
    os.mkdir(save_dir+'/test')
    with tqdm(total=3) as pbar:
        for i in ['ArmChairs', 'Chairs', 'Tables']:
            os.mkdir(save_dir+'/train/'+i)
            os.mkdir(save_dir+'/val/'+i)
            os.mkdir(save_dir+'/test/'+i)
            for j in os.listdir(save_dir+'/'+i):
                os.mkdir(save_dir+'/train/'+i+'/'+j)
                os.mkdir(save_dir+'/val/'+i+'/'+j)
                os.mkdir(save_dir+'/test/'+i+'/'+j)
                
                x = os.listdir(save_dir+'/'+i+'/'+j)
                random.shuffle(x)
                train, test = train_test_split(x, test_size=0.4)
                val, test = train_test_split(test, test_size=0.5)
                for k in train:
                    path = save_dir+'/train/'+i+'/'+j
                    Image.open(save_dir+'/'+i+'/'+j+'/'+k).save(path+'/'+k)
                for k in val:
                    path = save_dir+'/val/'+i+'/'+j
                    Image.open(save_dir+'/'+i+'/'+j+'/'+k).save(path+'/'+k)
                for k in test:
                    path = save_dir+'/test/'+i+'/'+j
                    Image.open(save_dir+'/'+i+'/'+j+'/'+k).save(path+'/'+k)
                    
            pbar.update(1)
    print("Done!")