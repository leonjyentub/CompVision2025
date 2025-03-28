import torch

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Example usage
if __name__ == "__main__":
    # Create a ResidualBlock
    block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
    
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(1, 64, 32, 32)
    
    # Forward pass through the block
    output = block(x)
    print(block)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    from torch.utils.tensorboard import SummaryWriter

    # 創建SummaryWriter實例
    writer = SummaryWriter('runs/model_visualization')

    # 將模型添加到TensorBoard
    writer.add_graph(block, x)
    writer.close()

    # 啟動TensorBoard
    # 在終端中運行: tensorboard --logdir=runs