import os
from PIL import Image
import matplotlib.pyplot as plt

def chooseCroppedImages(img1_cropped, img2_cropped, choice1, choice2, show=True):
  img_array_1 = []
  c = 0
  for i in choice1:
    img_array_1.append(img1_cropped[c][i])
    c+=1

  img_array_2 = []
  c = 0
  for i in choice2:
    img_array_2.append(img2_cropped[c][i])
    c+=1

  if show:
    fig1, axs1 = plt.subplots(1, len(img_array_1), figsize=(2*len(img_array_1), 2))
    # if len(choice1) == 1:
    for i, img in enumerate(img_array_1):
      axs1[i].set_title(str(i))
      axs1[i].imshow(img)
      # print(bboxes1[i])
    plt.subplots_adjust(wspace=0.4)
    plt.show()

    fig2, axs2 = plt.subplots(1, len(img_array_2), figsize=(2*len(img_array_2), 2))
    # if len(choice2) == 1:
    for i, img in enumerate(img_array_2):
      axs2[i].set_title(str(i))
      axs2[i].imshow(img)
    plt.subplots_adjust(wspace=0.4)
    plt.show()

  return img_array_1, img_array_2

# parse through Dataset and get image array
img_cropped_images = []
for i in os.listdir('./Dataset/orientation1'):
    x = os.listdir('./Dataset/orientation1/{}'.format(i))
    x = sorted(x, key = lambda x: int(x[:-4]))
    imgs = []
    for j in x:
        path = './Dataset/orientation1/{}/{}'.format(i, j)
        imgs.append(Image.open(path))
    img_cropped_images.append(imgs)
        

# indices of image that you want among each view in folder 'Dataset' for a particular orientation
complete_graph_indices = [0, 0, 0, 0, 0, 0, 0, 0]

img_array_1, img_array_2 = chooseCroppedImages(img_cropped_images, img_cropped_images, complete_graph_indices, complete_graph_indices)

try:
    os.mkdir('SelectedImages')
except:
    print("fail")

j=1
for i, img in enumerate(img_array_1):
  img.save('./SelectedImages/{}.png'.format(j))
  j+=1