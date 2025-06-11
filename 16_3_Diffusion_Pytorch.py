
# %%
import configparser
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers import DDPMScheduler, UNet2DModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from torch_snippets_local import simple_show, subplots

warnings.filterwarnings('ignore')
# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

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

# %%
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape, 'Labels:', y)
simple_show(torchvision.utils.make_grid(x)[0], cmap='Greys');

# %%
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

net = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=1,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 128, 256),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",   # a regular ResNet upsampling block
      ),
)

'''
DDPMScheduler 是 Denoising Diffusion Probabilistic Models(DDPM)的離散時序噪聲調度器
負責設定訓練及推論過程中的噪聲增添與去噪規則 
DDPMScheduler(num_train_timesteps=1000) 則是初始化一個 DDPMScheduler 實例
設定訓練過程中的時間步數為 1000。
'''
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def corrupt(xb, timesteps=None):
  if timesteps is None:
    timesteps = torch.randint(0, 999, (len(xb),)).long().to(device)
  noise = torch.randn_like(xb)
  noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
  return noisy_xb, timesteps

# %%
n_epochs = 1
#report = Report(n_epochs)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

scheduler = CosineAnnealingLR(opt, T_max=len(train_dataloader))
net = net.to(device)
for epoch in range(n_epochs):
    n = len(train_dataloader)
    for bx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)  # Data on the GPU
        noisy_x, timesteps = corrupt(x)  # Create our noisy x
        pred = net(noisy_x, timesteps).sample
        loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Batch {bx+1}/{n}, Loss: {loss.item():.4f}", end='\r')
    
# %%
net.cpu()
noise = torch.randn(5,1,32,32).to(net.device)
progress = [noise[:,0]]

for ts in np.logspace(np.log10(999), 0.1, 100):
  ts = torch.Tensor([ts]).long().to(net.device)
  noise = net(noise, ts).sample.detach().cpu()
  noise, _ = corrupt(noise, ts)
  progress.append(noise[:,0])

print(len(progress))
_n = 10
subplots(torch.stack(progress[::_n]).permute(1, 0, 2, 3).reshape(-1, 32, 32), nc=11, sz=(10,4))




