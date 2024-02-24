from object_memory import *

# get paths
lora_path = 'models/vit_finegrained_5x40_procthor.pt'
dataset_root = './condensed_procthor_images'

# get poses
# p = np.load(os.path.join(dataset_root, "pose.npy"))

# poses = np.zeros((8,7), dtype=np.float64)
# for i, pose in enumerate(p):
#     poses[i, :3] = pose[:3]
#     poses[i, 3:] = Rot.from_euler('xyz', pose[3:], degrees=True).as_quat()

# # list objects
# objects = [
#     "armchairs",
#     "chairs",
#     "coffee_tables",
#     "dining_tables",
#     "floor_lamps",
#     "garbage_cans",
#     "side_tables",
#     "sofas",
#     "tv_stands"
# ]

# accumulate paths
import os, sys
broad_ctg = [i for i in os.listdir(os.path.join(dataset_root, 'test'))]
print(broad_ctg)

instance_names = dict()
for b in broad_ctg:
    instance_names[b] = [i for i in os.listdir(os.path.join(dataset_root, 'test', b))]

all_img_paths = dict()
for b in sorted(instance_names):
    all_img_paths[b] = dict()
    for i in instance_names[b]:
        all_img_paths[b][i]  = [os.path.join(dataset_root, 'train', b, i, k) for k in os.listdir(os.path.join(dataset_root, 'train', b, i))]
        all_img_paths[b][i] += [os.path.join(dataset_root, 'test', b, i, k) for k in os.listdir(os.path.join(dataset_root, 'test', b, i))]
        all_img_paths[b][i] += [os.path.join(dataset_root, 'val', b, i, k) for k in os.listdir(os.path.join(dataset_root, 'val', b, i))]

# print(all_img_paths)

from sklearn.manifold import TSNE

print("Begin")
rev = LoraRevolver('cuda')
rev.load_lora_ckpt_from_file("/home2/aneesh.chavan/Change_detection/models/vit_finegrained_5x40_procthor.pt", name="tanay")

from tqdm import tqdm
from PIL import Image
import pickle

with open('/scratch/aneesh.chavan/armchairs_dataset_embs.pkl', 'rb') as file:
    pkldata = pickle.load(file)

data = []
labels = []

for b in sorted(pkldata):
    for i in sorted(pkldata[b]):
        for k in (pkldata[b][i]):
            # print(np.array(Image.open(k)).shape)
            # print(pkldata[b][i][k])
            data.append(k.cpu())
            labels.append(i)

print(data)
data = np.array(data)
data = data[:, 0, :]
print(data[0])#.shape)
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(data)

print("Dfas: ", embedded_data.shape)

# Plot the t-SNE results with different colors for each category
fig, ax = plt.subplots()
fig.set_size_inches(8,8)
# ax = plt.axes(projection ="3d")
conv = {label: i for i, label in enumerate(set([l for l in pkldata['armchairs']]))}
# print(conv)   

from mpl_toolkits import mplot3d
# scatter = ax.scatter3D(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], c=[conv[i] for i in labels], cmap='viridis', marker='o', edgecolor='w', s=50)
scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=[conv[i] for i in labels], cmap='viridis', marker='o', edgecolor='w', s=20)


# print(scatter)

plt.title('t-SNE Plot')
# plt.colorbar(scatter, label='Digit Class')
print([i for i in conv])

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

plt.savefig('./tsne_colored3.png')

exit(0)

with tqdm(total=sum(len(all_img_paths[b][i]) for b in instance_names for i in instance_names[b])) as pbar:
    all_embs = dict()
    for b in sorted(instance_names):
        all_embs[b] = dict()
        for i in instance_names[b]:
            all_embs[b][i] = []
            for k in (all_img_paths[b][i]):
                # print(np.array(Image.open(k)).shape)
                all_embs[b][i].append(rev.encode_image([np.array(Image.open(k))[..., :3]]))
                pbar.update(1)

        break
# print(all_embs[b][i])
# print(all_embs[b][i][0].shape)

print("dumping to %s" % ('/scratch/aneesh.chavan/%s_dataset_embs.pkl' % b))
with open('/scratch/aneesh.chavan/%s_dataset_embs.pkl' % b, 'wb') as file:
    pickle.dump(all_embs, file)