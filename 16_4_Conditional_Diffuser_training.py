# %%
import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset

from torch_snippets_local import flatten, subplots

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

config = configparser.ConfigParser()
config.read('config.ini')
# Load parameters from config file
#root = '/Users/leonjye/Documents/MachineLearingData'
root = config.get('DEFAULT', 'root_dir')

# %%
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)

batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, labels):
        return self.embedding(labels)

embedding_layer = EmbeddingLayer(num_embeddings=10, embedding_dim=32).to(device)

class ConditionalUNet2DModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels += 32  # Adjust for embedding dimension

net = ConditionalUNet2DModel(
    sample_size=28,
    in_channels=1 + 32,  # 1 for original channel, 20 for embedding
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(32, 64, 128, 256),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to(device)

# %%
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def corrupt_with_embedded_labels(xb, labels, timesteps=None):
    if timesteps is None:
        timesteps = torch.randint(0, 999, (len(xb),)).long().to(device)
    noise = torch.randn_like(xb)
    noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
    labels_embedded = embedding_layer(labels).unsqueeze(-1).unsqueeze(-1)
    labels_embedded = labels_embedded.expand(-1, -1, xb.shape[2], xb.shape[3])
    return torch.cat([noisy_xb, labels_embedded], dim=1), timesteps

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_dataloader))

n_epochs = 3
#report = Report(n_epochs)

# %%
for epoch in range(n_epochs):
    n = len(train_dataloader)
    for bx, (x, labels) in enumerate(train_dataloader):
        x = x.to(device)
        labels = labels.to(device)
        noisy_x, timesteps = corrupt_with_embedded_labels(x, labels)
        pred = net(noisy_x, timesteps).sample # Exclude the embedding channels from the prediction
        loss = loss_fn(pred, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{n_epochs}], Batch [{bx + 1}/{n}], Loss: {loss.item():.4f}', end='\r')

# %%
x.shape

# %%
xb = torch.zeros(10, 1, 32, 32)

# %%
timesteps = torch.randint(999, 1000, (len(xb),)).long().to(device)

# %%
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps).to(device)

# %%
labels = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device)  # Labels for which you want to generate images

# %%
labels_embedded = embedding_layer(labels).unsqueeze(-1).unsqueeze(-1)
labels_embedded = labels_embedded.expand(-1, -1, xb.shape[2], xb.shape[3]).to(device)

# %%
noisy_x = torch.cat([noisy_xb, labels_embedded], dim=1)

# %%
pred = net(noisy_x, timesteps).sample.permute(0,2,3,1).reshape(-1, 32, 32)

# %%
subplots(pred.detach().cpu().numpy())

# %%
net.to(device)
labels = torch.Tensor([0, 1, 2, 3, 4]).long().to(device)  # Labels for which you want to generate images
embeddings = embedding_layer(labels)  # Generate embeddings for each label
embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)  # Resize to match the spatial dimensions of the noise

noise = torch.randn(labels.size(0), 1, 32, 32).to(device)  # Generate initial noise
progress = [noise]

for ts in np.logspace(np.log10(999), 0.1, 100):
    ts_tensor = torch.Tensor([ts]).long().expand(labels.size(0)).to(device)
    if ts>998:
      combined_input = torch.cat([noise, embeddings], dim=1)  # Combine noise and label embeddings
    else:
      combined_input = noise
    noise = net(combined_input, ts_tensor).sample.detach()  # Generate image conditioned on label
    #noise = noise[:, :-20, :, :]  # Exclude the embedding channels from the output
    progress.append(noise)
    # Recorrupt the noise for the next step
    noise, _ = corrupt_with_embedded_labels(noise, labels, ts_tensor)

print(len(progress))
_n = 20
subplots(
    torch.cat(progress, dim=1)[:,::_n].permute(1, 0, 2, 3).reshape(-1, 32, 32),
    nc=5,
    sz=(4, 10),
    titles=flatten([[int(i)]*len(labels) for i in np.logspace(np.log10(999), 0.1, 101)[::_n]])
)
plt.tight_layout()

# %%
plt.savefig('16_4_Conditional_Diffuser_training.png', dpi=300, bbox_inches='tight')