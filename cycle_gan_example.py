
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 定義生成器（簡化 UNet 結構）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

# 定義判別器（PatchGAN 結構）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

# 初始化網路
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

# 損失函數
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizer
lr = 0.0002
optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# 資料轉換
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 假資料集
dataset_A = datasets.ImageFolder("data/trainA", transform=transform)
dataset_B = datasets.ImageFolder("data/trainB", transform=transform)
loader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
loader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

# 訓練循環 (簡化版)
for epoch in range(1):
    for (real_A, _), (real_B, _) in zip(loader_A, loader_B):
        # 對抗損失
        valid = torch.ones((1, 1, 31, 31))
        fake = torch.zeros((1, 1, 31, 31))

        # Generator 前向
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        # Cycle 損失
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        # Identity 損失
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        # Generator 總損失
        loss_G = loss_GAN_AB + loss_GAN_BA + 10 * (loss_cycle_A + loss_cycle_B) + 5 * (loss_id_A + loss_id_B)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Discriminator A
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B
        loss_real_B = criterion_GAN(D_B(real_B), valid)
        loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

    print(f"Epoch {epoch}: Generator Loss: {loss_G.item()}")

