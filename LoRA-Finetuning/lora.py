from lora_setup import *

batch_size = 8

train_dataset = ObjectTriplets('/scratch/sarthak.chittawar/LoRA/train', train_transforms, num_triplets_per_class=140, difficult_triplet_percentage=0.5)
val_dataset = ObjectTriplets('/scratch/sarthak.chittawar/LoRA/val', val_transforms, num_triplets_per_class=40, difficult_triplet_percentage=0.5)
test_dataset = ObjectTriplets('/scratch/sarthak.chittawar/LoRA/test', test_transforms, num_triplets_per_class=20, difficult_triplet_percentage=0.5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# returns embeddings for anchor, positive and negative images
def get_embeddings(model, batch):
    # get cls tokens directly
    a_embs = model(batch[:, 0, ...], output_hidden_states=True).last_hidden_state[:,0,:]
    p_embs = model(batch[:, 1, ...], output_hidden_states=True).last_hidden_state[:,0,:]
    n_embs = model(batch[:, 2, ...], output_hidden_states=True).last_hidden_state[:,0,:]

    return a_embs, p_embs, n_embs

def get_triplet_loss(a_embs, p_embs, n_embs, bias_val=0.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bias = torch.ones(a_embs.shape[0]).float() * bias_val
    bias = bias.to(device)

    z = torch.zeros(a_embs.shape[0]).float().to(device)

    loss = torch.max(z,
               (
                  bias +
                  torch.linalg.norm(a_embs - p_embs, dim=1) -
                  torch.linalg.norm(a_embs - n_embs, dim=1)
               ))

    loss = torch.sum(loss)

    return loss


class train_config():
  def __init__(self,
               r=16,
               bias=1.0,
               batch_size=16,
               num_epochs=10,
               ):
    self.r = r
    self.bias = bias
    self.batch_size = batch_size
    self.num_epochs = num_epochs

def train(model, train_loader, val_loader, optimizer, config):
    """
    Generic training loop for PyTorch models.

    Args:
    model: The neural network model to train.
    train_loader: DataLoader for the training dataset.
    criterion: The loss function (e.g., nn.CrossEntropyLoss).
    optimizer: The optimizer (e.g., optim.SGD or optim.Adam).
    device: The device (CPU or GPU) to perform training.
    num_epochs: The number of training epochs.

    Returns:
    None
    """
    model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(config.num_epochs):
        print("Epoch: ", epoch+1)
        running_loss = 0.0

        for triplet in tqdm(train_loader):
            triplet = triplet.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            a,p,n = get_embeddings(lora_model, triplet)
            loss = get_triplet_loss(a,p,n, config.bias)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)

        # Calculate validation loss
        val_running_loss = 0.0
        with torch.no_grad():
          for val_triplet in val_loader:
            val_triplet = val_triplet.to(device)
            a,p,n = get_embeddings(lora_model, val_triplet)
            val_loss = get_triplet_loss(a,p,n, config.bias)

            val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)


        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Loss: {epoch_loss:.4f} - Validation loss: {val_epoch_loss:.4}")
        print()

    print("Training complete.")

# Example usage:
# train(model, train_loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), "cuda", num_epochs=10)

optimizer = optim.Adam(lora_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

train_cfg = train_config(bias=2.5)
train(lora_model, train_loader, val_loader, optimizer, train_cfg)

# Visualise test set images
test_path =  '/scratch/sarthak.chittawar/LoRA/test'

categories = os.listdir(test_path)
test_classes = []
for ctg in categories:
  for i in os.listdir(os.path.join(test_path, ctg)):
    test_classes.append(ctg + '/' + i)

test_classes = sorted(test_classes)

test_images = [
    os.listdir(os.path.join(
        test_path,
        c
    )) for c in test_classes
]

for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = os.path.join(test_path, test_classes[i], test_images[i][j])
    
for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = PIL.Image.open(test_images[i][j])

# to_show = 10
# fig, axes = plt.subplots(len(test_classes), to_show, figsize=(to_show * 2.5, len(test_classes) * 2.5))

# for i in tqdm(range(len(test_classes))):
#   for j, img in enumerate(test_images[i][:to_show]):
#     ax = axes[i][j]
#     ax.imshow(img)
#     ax.axis('off')  # Hide axis

# plt.show()

# get all embeddings

w = []

with torch.no_grad():
  for row in test_images:
    r = []
    for i in row[:18]:
      im = test_transforms(i)
      k = lora_model(im.unsqueeze(0).cuda(), output_hidden_states=True).last_hidden_state[0,0,:]

      r.append(k)
    w.append(torch.stack(r))
w = torch.stack(w).reshape(-1, 768)

# get heatmap

scores = (torch.nn.functional.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1)).detach().cpu().numpy()

plt.imshow(scores, cmap='hot')
plt.colorbar()

x_axis_titles = [f"{test_classes[i//10]}" for i in range(5, 5 + len(scores), 10)]
y_axis_titles = [f"{test_classes[i//10]}" for i in range(5, 5 + len(scores), 10)]

plt.xticks(range(5, 5 + len(scores), 10), x_axis_titles, rotation=45, ha='right')
plt.yticks(range(5, 5 + len(scores), 10), y_axis_titles, va='center')

for i in range(1, len(scores)):
    if i % 10 == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=1.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=1.5)


# Show the heatmap
plt.title("Base ViT vision encoder, no finetuning")

plt.show()