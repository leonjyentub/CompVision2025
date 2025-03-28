import os
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image

class UTKFaceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Custom Dataset for UTKFace dataset
        
        Args:
            image_paths (list): List of image file paths
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 從檔名解析標籤
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        
        # 解析檔名 [age]_[gender]_[race]_[date&time].jpg
        try:
            age, gender, race, _ = filename.split('_')
        except:
            age, gender, _ = filename.split('_') #有可能只有age和gender
        # 讀取圖片
        image = Image.open(image_path).convert('RGB')
         
        # 應用transform
        if self.transform:
            image = self.transform(image)
        
        # 轉換標籤為張量
        age = torch.tensor(int(age), dtype=torch.float32)
        gender = torch.tensor(int(gender), dtype=torch.long)
        
        return image, (age, gender)

class AgeGenderClassificationModel(nn.Module):
    def __init__(self, num_age_classes=117, num_gender_classes=2):
        """
        多任務學習模型，基於VGG16
        
        Args:
            num_age_classes (int): 年齡類別數量
            num_gender_classes (int): 性別類別數量
        """
        super(AgeGenderClassificationModel, self).__init__()
        
        # 載入預訓練VGG16模型
        vgg16 = torchvision.models.vgg16(weights='DEFAULT')
        
        # 凍結特徵提取層的參數
        for param in vgg16.features.parameters():
            param.requires_grad = False
        
        # 自定義分類器
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        # 多任務輸出層
        self.age_head = nn.Linear(1024, num_age_classes)
        self.gender_head = nn.Linear(1024, num_gender_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        age_output = self.age_head(x)
        gender_output = self.gender_head(x)
        
        return age_output, gender_output

def train_and_validate(model, train_loader, val_loader, criterion_age, criterion_gender, optimizer, device, writer, num_epochs=10):
    """
    訓練和驗證模型，使用TensorBoard記錄
    
    Args:
        model (nn.Module): 深度學習模型
        train_loader (DataLoader): 訓練數據加載器
        val_loader (DataLoader): 驗證數據加載器
        criterion_age (nn.Module): 年齡損失函數
        criterion_gender (nn.Module): 性別損失函數
        optimizer (torch.optim): 優化器
        device (torch.device): 計算設備
        writer (SummaryWriter): TensorBoard寫入器
        num_epochs (int): 訓練輪數
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_age_loss, train_gender_loss = 0.0, 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            age_labels, gender_labels = labels[0].to(device), labels[1].to(device)
            
            optimizer.zero_grad()
            
            age_outputs, gender_outputs = model(images)
            
            age_loss = criterion_age(age_outputs, age_labels.long())
            gender_loss = criterion_gender(gender_outputs, gender_labels)
            
            total_loss = age_loss + gender_loss
            total_loss.backward()
            optimizer.step()
            
            train_age_loss += age_loss.item()
            train_gender_loss += gender_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}')
                print(f'Train Loss - Age: {age_loss.item():.4f}, Gender: {gender_loss.item():.4f}')
        
        # 驗證階段
        model.eval()
        val_age_loss, val_gender_loss = 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                age_labels, gender_labels = labels[0].to(device), labels[1].to(device)
                
                age_outputs, gender_outputs = model(images)
                
                age_loss = criterion_age(age_outputs, age_labels.long())
                gender_loss = criterion_gender(gender_outputs, gender_labels)
                
                val_age_loss += age_loss.item()
                val_gender_loss += gender_loss.item()
        
        # 計算平均損失
        train_age_loss /= len(train_loader)
        train_gender_loss /= len(train_loader)
        val_age_loss /= len(val_loader)
        val_gender_loss /= len(val_loader)
        
        # TensorBoard記錄
        writer.add_scalar('Train/Age Loss', train_age_loss, epoch)
        writer.add_scalar('Train/Gender Loss', train_gender_loss, epoch)
        writer.add_scalar('Validation/Age Loss', val_age_loss, epoch)
        writer.add_scalar('Validation/Gender Loss', val_gender_loss, epoch)
        
        # 保存最佳模型
        if val_age_loss + val_gender_loss < best_val_loss:
            best_val_loss = val_age_loss + val_gender_loss
            torch.save(model.state_dict(), '05_AgeGender_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss - Age: {train_age_loss:.4f}, Gender: {train_gender_loss:.4f}')
        print(f'Val Loss - Age: {val_age_loss:.4f}, Gender: {val_gender_loss:.4f}')

def main():
    # 設置超參數和設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 0.001
    
    # 定義數據轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 收集資料集路徑
    dataset_path = 'data/UTKFace'
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    
    # 分割訓練和驗證集
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    
    # 創建數據加載器
    train_dataset = UTKFaceDataset(train_paths, transform=transform)
    val_dataset = UTKFaceDataset(val_paths, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型和損失函數
    model = AgeGenderClassificationModel().to(device)
    criterion_age = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # TensorBoard寫入器
    writer = SummaryWriter('runs/age_gender_classification')
    
    # 訓練和驗證
    train_and_validate(model, train_loader, val_loader, 
                       criterion_age, criterion_gender, 
                       optimizer, device, writer)
    writer.add_graph(model, torch.randn(1, 3, 224, 224).to(device))
    # 關閉寫入器
    writer.close()

if __name__ == '__main__':
    main()