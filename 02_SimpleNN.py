import torch                  # PyTorch 的核心庫
import torch.nn as nn         # 神經網路模組
import torch.optim as optim     # 優化器模組
from torchvision import datasets, transforms # 資料集和轉換工具
from torch.utils.data import DataLoader # 資料載入器 (Dataset -> DataLoader) , 用于批量处理数据以供模型训练。通常用于将数据集分成小批量(batch),并提供给模型进行训练。
import matplotlib.pyplot as plt # 可視化工具 (如果需要)
import torch
import numpy as np

# 1. 定義神經網路模型 (Model) : 使用 nn.Module 建立一個類別
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()  # 繼承 nn.Module 的初始化方法
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一個全連接層
        self.batch_norm = nn.BatchNorm1d(hidden_size) # Batch Normalization 層
        self.dropout = nn.Dropout(p=0.5)              # Dropout 層
        self.relu = nn.ReLU()                        # ReLU 激活函數
        self.fc2 = nn.Linear(hidden_size, num_classes) # 第二個全連接層 (輸出層)

    def forward(self, x): # 定義前向傳播的過程
        out = self.fc1(x)       # 輸入通過第一個全連接層
        out = self.batch_norm(out) # 通過 Batch Normalization 層
        out = self.relu(out)      # 經過 ReLU 激活函數
        out = self.dropout(out)   # 通過 Dropout 層
        out = self.fc2(out)       # 通過第二個全連接層 (輸出層)
        return out

# 2. 設定超參數 (Hyperparameters):  決定模型訓練的細節, 例如學習率、批量大小等.
input_size = 784          # 輸入資料的維度 (例如: MNIST 資料集的圖片大小是 28x28=784)
hidden_size = 128         # 隱藏層的大小
num_classes = 10          # 輸出的類別數量 (例如: MNIST 資料集有 10 個數字)
learning_rate = 0.001     # 學習率,控制每次更新權重的幅度
batch_size = 64           # 批次大小,每次訓練使用的樣本數量
num_epochs = 5            #  訓練的輪數,整個資料集要被模型學習多少次。

# 確認是否有可用的 GPU , 如果有, 使用 GPU 加速計算, 如果沒有則使用 CPU .
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 確認是否有可用的 GPU, 如果有則使用 GPU , 如果沒有則使用 CPU

# 將模型移動到指定的設備上 (GPU 或 CPU).  如果使用了 GPU, 模型和資料都會被移動到 GPU 上进行计算。
model = SimpleNN(input_size, hidden_size, num_classes).to(device)

# 定義損失函數 (Loss function):  衡量模型的預測與真實標籤之間的差異。
criterion = nn.CrossEntropyLoss() # 使用交叉熵損失函數,適用於多分類問題

# 定義優化器 (Optimizer): 
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
# 使用隨機梯度下降 (SGD) 優化器, lr 是學習率, weight_decay 是 L2 正則化的權重。

# 加載資料集:MNIST 数据集加载和预处理。
transform = transforms.Compose([  # 定义数据转换操作(transforms):将图像转换为张量并进行标准化。这个过程是在加载数据集的时候进行的。
    transforms.ToTensor(),         # 将图像转换为 PyTorch 张量,将像素值从 [0, 255]  归一化到 [0,1] 的范围。
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化: 用均值和标准差对图像进行标准化处理。(MNIST 資料集的均值和標準差). 使數據的分布更接近標準正態分佈,有助于模型训练。这通常可以加快收敛速度并提高模型的性能。
])
transform = transforms.Compose([  
    transforms.ToTensor(),
    lambda x: x*255
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # 加载训练数据集: 从 torchvision 的 datasets 中加载 MNIST 数据集。 root 指定数据集的存储位置; train=True 表明加载的是训练集;download=True 表示如果本地没有数据集则下载;transform 应用上述定义的转换操作到数据集中的每个样本上。它会将图片转换为张量并进行标准化处理 。 这使得模型更容易学习特征并且减少了梯度消失或爆炸的可能性。所以说transform 是很重要的一步! 有些时候我们需要针对特定的问题自己定义 transform ,比如说灰度图转彩色图等等!具体的使用方法需要根据实际情况来定!后续我们会继续讲解相关的知识点! 请大家持续关注!
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # 加载测试数据集

# 输出第一个样本的numpy陣列的形状
print(np.array(train_dataset[0][0]).shape)
print(np.array(train_dataset[0][0]))
print(train_dataset[0])
print(train_dataset[0][1]) # 输出第一个样本的标签

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # 创建训练数据加载器: 使用 DataLoader 来批量加载训练数据。 batch_size 指定每个批次的大小,shuffle=True 表示在每个 epoch 重新打乱数据顺序。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)   # 创建测试数据加载器

losses, accuracies = [], []
# 3. 訓練模型 (Train the model):
total_step = len(train_loader) # 计算训练数据的总步数,一个 step 指的是一个批次的训练过程。
for epoch in range(num_epochs): # 外循环是 epoch 的循环,表示对整个数据集进行多次迭代。
    epoch_losses = []
    for i, (images, labels) in enumerate(train_loader): # 内循环是遍历每个批次的数据。 images 是一个 batch 的图像,labels 是对应的标签。enumerate 用于同时获取索引和数据。
        # 将图像和标签移动到指定的设备上 (GPU 或 CPU).  如果使用了 GPU, 模型和資料都會被移動到 GPU 上进行计算。
        images = images.reshape(-1, 28*28).to(device) # 将图像进行 reshape 操作,将其转换为适合输入到全连接层的形状(batch_size, input_size)。
        labels = labels.to(device)  # 将标签也移动到指定的设备上

        # 前向传播 (Forward pass): 将输入数据传递给模型,得到输出结果。
        outputs = model(images)  # 调用模型进行前向传播,得到预测的输出值。
        loss = criterion(outputs, labels) # 计算损失: 使用损失函数计算模型的预测结果与真实标签之间的差异。
        epoch_losses.append(loss.item()) # 将每个批次的损失值添加到 epoch_losses 列表中。
        # 反向传播和優化 (Backward and optimize): 根据损失计算梯度,并更新模型的权重.
        optimizer.zero_grad()   # 梯度清零: 在进行反向传播之前,需要将优化器的梯度清零,因为 PyTorch 中梯度的累加的。
        loss.backward()         # 反向传播:  计算损失相对于模型参数的梯度。
        optimizer.step()        # 更新参数:  使用优化器根据计算出的梯度来更新模型的参数,从而降低损失值。

        if (i+1) % 100 == 0: # 每 100 个批次打印一次训练信息, 显示当前的训练进度和损失值
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' # 输出当前 epoch、step、总步数以及 loss 值。 .format 是格式化字符串的方法。
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item())) # loss.item() 获取的是 tensor 中的数值
    model.eval()  # 將模型設置為評估模式 (用於推理/測試)。在測試階段禁用 dropout 和 batch normalization 等訓練時使用的特定行為。這確保了測試結果的一致性 。 注意:在使用某些模組(如 Dropout 和 BatchNorm)時,必須將模型切換到評估模式 。 否则会影响测试的结果! 因为这些模块在训练和测试阶段的行为是不同的。
    with torch.no_grad(): # 禁用梯度計算,以節省內存和加快計算速度。在測試階段不需要計算梯度。
        correct = 0         #  用於統計預測正確的樣本數量。
        total = 0           #  用於統計總樣本數量。
        for images, labels in test_loader: # 遍历测试集中的每个批次数据。
            images = images.reshape(-1, 28*28).to(device) # 将图像进行 reshape 操作,使其形状与模型输入匹配。
            labels = labels.to(device)   # 将标签也移动到指定的设备上

            outputs = model(images)       # 前向传播,得到模型的输出结果 (预测值)。
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果: 使用 torch.max() 函数找到每个样本中概率最大的类别作为预测结果。 _ 表示忽略最大值的索引 (我们只需要最大值)。 .data 用于获取输出张量中的数据,1 表示在 dim=1 上找到最大值。
            total += labels.size(0)                 #  统计总样本数量:  计算当前批次的总样本数量并累加到 total 变量中。
            correct += (predicted == labels).sum().item() # 统计预测正确的样本数量: 将预测结果与真实标签进行比较,计算当前批次预测正确的样本数量,并累加到 correct 变量中。
        
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) # 计算并打印模型在测试集上的准确率.
    losses.append(np.mean(epoch_losses)) # 将每个 epoch 的损失值添加到 losses 列表中。
    accuracies.append(100 * correct / total) # 将每个 epoch 的准确率添加到 accuracies 列表中。
# 5. 保存模型 (Save the model): 将训练好的模型保存到磁盘上,以便后续使用。
torch.save(model.state_dict(), 'model.pth') # 保存模型參數: .state_dict() 獲取模型的參數 (權重和偏置),'model.pth' 是保存的文件名。

print("訓練完成!")

epochs = np.arange(5)+1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.show()