import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np

# GPU 設定與檢測
def setup_device():
    """設定並返回可用的計算裝置(GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_count = torch.cuda.device_count()
        device_capability = torch.cuda.get_device_capability(current_device)
        print(f"GPU 可用！使用: {device_name}")
        print(f"設備數量: {device_count}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"設備能力: {device_capability}")
        print(f"目前顯存使用量: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
        print(f"最大顯存使用量: {torch.cuda.max_memory_allocated(current_device) / 1024**2:.2f} MB")
    else:
        device = torch.device("cpu")
        print("GPU 不可用，使用 CPU 進行計算")
    
    return device

# 設定隨機種子以確保結果可重現
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 對於固定輸入大小，可以設為True以提高性能

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一個卷積層
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二個卷積層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut連接
        self.shortcut = nn.Sequential()
        
        # 如果輸入和輸出的維度不同（通道數或尺寸），則需要調整shortcut的維度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 儲存原始輸入，用於後續的殘差連接
        identity = x
        
        # 前向傳播的主路徑
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 將shortcut（恆等映射或經過調整）與主路徑輸出相加
        out += self.shortcut(identity)
        
        # 最後應用ReLU激活函數
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # 降維卷積
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷積
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 升維卷積
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # shortcut連接
        self.shortcut = nn.Sequential()
        
        # 如果輸入和輸出的維度不同，則需要調整shortcut
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x
        
        # 前向傳播的主路徑
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 將shortcut與主路徑輸出相加
        out += self.shortcut(identity)
        
        # 應用ReLU激活
        out = F.relu(out)
        
        return out


class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        
        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 創建殘差層
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全連接層
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始層
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 殘差層
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 全局平均池化
        out = F.avg_pool2d(out, 4)
        
        # 展平特徵
        out = out.view(out.size(0), -1)
        
        # 分類層
        out = self.linear(out)
        
        return out


# 創建ResNet-18模型
def ResNet18():
    return SimpleResNet(BasicBlock, [2, 2, 2, 2])

# 創建ResNet-50模型
def ResNet50():
    return SimpleResNet(Bottleneck, [3, 4, 6, 3])


# 追蹤GPU顯存使用的方法
def log_gpu_memory_usage(writer, epoch, device=None):
    """將GPU顯存使用情況記錄到TensorBoard"""
    if not torch.cuda.is_available():
        return
    
    current_device = torch.cuda.current_device() if device is None else device
    allocated = torch.cuda.memory_allocated(current_device) / 1024**2  # MB
    cached = torch.cuda.memory_reserved(current_device) / 1024**2      # MB
    
    writer.add_scalar('GPU/Memory_Allocated_MB', allocated, epoch)
    writer.add_scalar('GPU/Memory_Cached_MB', cached, epoch)
    
    return allocated, cached


# 準備CIFAR-10數據集
def get_cifar10_data_loaders(batch_size=128, num_workers=4, pin_memory=True):
    """
    加載CIFAR-10數據集並優化GPU數據傳輸
    
    參數:
        batch_size: 每個批次的大小
        num_workers: DataLoader使用的工作進程數（針對CPU）
        pin_memory: 是否將數據固定在分頁鎖定內存中，以加速CPU->GPU的傳輸
    
    返回:
        trainloader, testloader: 訓練和測試數據加載器
    """
    # 轉換並增強訓練數據
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 測試數據只需要標準化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加載數據集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory)

    return trainloader, testloader


# 訓練函數
def train(model, trainloader, optimizer, criterion, epoch, writer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 為了更準確的計時, 使用torch.cuda.synchronize()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清除之前的梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(inputs)
        
        # 計算損失
        loss = criterion(outputs, targets)
        
        # 反向傳播
        loss.backward()
        
        # 更新參數
        optimizer.step()
        
        # 統計
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每100個批次記錄一次
        if batch_idx % 100 == 0:
            # 記錄當前GPU使用情況
            if device.type == 'cuda':
                current_mem, _ = log_gpu_memory_usage(writer, epoch * len(trainloader) + batch_idx)
                mem_info = f"| GPU: {current_mem:.1f}MB"
            else:
                mem_info = ""
                
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}% {mem_info}')
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_time = time.time() - start_time
    
    # 記錄到TensorBoard
    train_loss = train_loss / len(trainloader)
    train_acc = 100. * correct / total
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Time/epoch_training_seconds', epoch_time, epoch)
    
    # 記錄所有層的權重和偏置的分佈
    for name, param in model.named_parameters():
        if 'bn' not in name:  # 不記錄BatchNorm層的參數
            writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:  # 有些參數可能沒有梯度
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    return train_loss, train_acc


# 測試函數
def test(model, testloader, criterion, epoch, writer, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向傳播
            outputs = model(inputs)
            
            # 計算損失
            loss = criterion(outputs, targets)
            
            # 統計
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    test_time = time.time() - start_time
    
    # 記錄到TensorBoard
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('Time/epoch_testing_seconds', test_time, epoch)
    
    # 如果使用GPU，記錄GPU顯存使用情況
    if device.type == 'cuda':
        log_gpu_memory_usage(writer, epoch)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | Time: {test_time:.2f}s')
    
    return test_loss, test_acc


# 將模型移至多GPU（如果可用）
def setup_model_for_gpu(model, device):
    """
    根據可用GPU數量設置模型，支持單GPU或多GPU
    """
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 個GPU!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    return model


# 主函數：訓練和測試模型，使用TensorBoard可視化
def run_tensorboard_demo(model_name="ResNet18", epochs=10, batch_size=128, 
                         learning_rate=0.1, momentum=0.9, weight_decay=5e-4,
                         num_workers=4):
    # 設定隨機種子確保可重現性
    set_seed(42)
    
    # 設定計算裝置
    device = setup_device()
    
    # 優化數據加載參數
    pin_memory = True if device.type == 'cuda' else False
    
    # 創建日誌目錄
    log_dir = f'runs/{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)
    
    # 獲取數據
    trainloader, testloader = get_cifar10_data_loaders(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # 創建模型
    if model_name == "ResNet18":
        model = ResNet18()
    elif model_name == "ResNet50":
        model = ResNet50()
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 設置模型到適當的設備
    model = setup_model_for_gpu(model, device)
    
    # 添加模型結構到TensorBoard
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    # 如果使用DataParallel，需要使用model.module來獲取原始模型
    if isinstance(model, nn.DataParallel):
        writer.add_graph(model.module, dummy_input)
    else:
        writer.add_graph(model, dummy_input)
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                          momentum=momentum, weight_decay=weight_decay)
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 記錄配置信息
    config = {
        'model': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'device': str(device),
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # 將配置添加到TensorBoard
    config_text = '\n'.join(f'{k}: {v}' for k, v in config.items())
    writer.add_text('Configuration', config_text)
    
    print(f"開始訓練 {model_name} 於 {device} 裝置...")
    
    # 訓練循環
    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch {epoch}/{epochs} -----")
        
        # 訓練與測試
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, epoch, writer, device)
        test_loss, test_acc = test(model, testloader, criterion, epoch, writer, device)
        
        # 更新學習率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # 儲存模型
        if epoch % 5 == 0 or epoch == epochs:
            checkpoint_path = f'{log_dir}/checkpoint_epoch_{epoch}.pth'
            
            # 若使用DataParallel, 保存 model.module 而非 model
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, checkpoint_path)
            print(f"已保存檢查點至 {checkpoint_path}")
    
    # 關閉SummaryWriter
    writer.close()
    
    print(f"訓練完成！TensorBoard日誌已保存到 {log_dir}")
    print("可以使用以下命令啟動TensorBoard：")
    print(f"tensorboard --logdir={log_dir}")
    
    return model


# 可視化特徵圖的函數
def visualize_feature_maps(model, sample_image, writer, layer_name="layer1", device=None):
    """
    可視化模型中特定層的特徵圖
    
    參數:
        model: 要可視化的模型
        sample_image: 輸入圖像
        writer: TensorBoard SummaryWriter
        layer_name: 要可視化的層名稱
        device: 計算裝置
    """
    if device is None:
        device = next(model.parameters()).device
    
    # 如果模型使用DataParallel，獲取基礎模型
    if isinstance(model, nn.DataParallel):
        model_for_hook = model.module
    else:
        model_for_hook = model
    
    # 註冊鉤子來獲取特定層的特徵圖
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 為目標層註冊前向鉤子
    if hasattr(model_for_hook, layer_name):
        getattr(model_for_hook, layer_name).register_forward_hook(get_activation(layer_name))
    
    # 確保模型處於評估模式
    model.eval()
    
    # 添加批次維度並移動到正確的設備
    sample_image = sample_image.unsqueeze(0).to(device)
    
    # 進行前向傳播
    with torch.no_grad():
        model(sample_image)
    
    # 獲取激活值
    if layer_name in activation:
        feature_maps = activation[layer_name]
        
        # 選擇前16個特徵圖可視化
        num_maps = min(16, feature_maps.size(1))
        for i in range(num_maps):
            feature_map = feature_maps[0, i].cpu().numpy()
            writer.add_image(f'{layer_name}/feature_map_{i}', 
                            feature_map.reshape(1, feature_map.shape[0], feature_map.shape[1]), 
                            global_step=0, 
                            dataformats='CHW')
    else:
        print(f"找不到層: {layer_name}")


# 示範如何使用TensorBoard可視化的主函數
def tensorboard_demo(use_gpu=True):
    """
    進行TensorBoard可視化示範
    
    參數:
        use_gpu: 是否使用GPU (若可用)
    """
    # 設定隨機種子
    set_seed(42)
    
    # 檢測GPU
    device = setup_device() if use_gpu else torch.device("cpu")
    
    # 選擇適合的批次大小
    # 較大的GPU記憶體可以使用較大的批次大小
    if device.type == 'cuda':
        batch_size = 256  # 可以根據實際GPU顯存適當調整
        num_workers = 4   # GPU通常可以處理更多的worker
    else:
        batch_size = 64   # CPU通常使用較小的批次大小
        num_workers = 2   # CPU通常使用較少的worker
    
    # 初始化模型並進行簡單訓練
    # 使用較小的epochs以快速演示
    model = run_tensorboard_demo(
        model_name="ResNet50",
        epochs=50,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # 創建一個新的writer，專門用於可視化特徵圖
    feature_writer = SummaryWriter('runs/feature_maps_ResNet50')
    
    # 獲取一個樣本圖像用於特徵圖可視化
    _, testloader = get_cifar10_data_loaders(batch_size=1)
    sample_data = next(iter(testloader))[0][0]  # 獲取第一個樣本的圖像
    
    # 可視化不同層的特徵圖
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        visualize_feature_maps(model, sample_data, feature_writer, layer_name, device)
    
    feature_writer.close()
    
    print("特徵圖可視化完成！可以使用以下命令啟動TensorBoard來查看：")
    print("tensorboard --logdir=runs")


if __name__ == "__main__":
    tensorboard_demo(use_gpu=True)  # 設置為True使用GPU (如果可用)
