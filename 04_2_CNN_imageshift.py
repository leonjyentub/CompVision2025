import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Data preprocessing and loading
transform_affine = transforms.Compose([
    transforms.ToTensor(),
    #進行圖片的小幅度平移
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.Normalize((0.5,), (0.5,))
])

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#一樣的訓練集，但測試集不一樣，一個測試集有transform，另一個測試集沒有transform
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset_affine = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_affine, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#兩個CNN模型，一個透過maxpooling，另一個沒有，測試pooling的效果，是否能對平移有更好的辨識
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class CNN_no_pooling(nn.Module):
    def __init__(self):
        super(CNN_no_pooling, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.fc = nn.Linear(28*28*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
#訓練模型
def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        #print accuracy for each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    return model

#測試模型
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return

#訓練模型
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader_affine = torch.utils.data.DataLoader(dataset=test_dataset_affine, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
model = CNN()
model = model.to(device)
model = train_model(model, train_loader)
print('CNN with maxpooling')
test_model(model, test_loader_affine)
test_model(model, test_loader)
print('-'*50)
print('CNN without maxpooling')
model_no_pooling = CNN_no_pooling()
model_no_pooling = model_no_pooling.to(device)
model_no_pooling = train_model(model_no_pooling, train_loader)
test_model(model_no_pooling, test_loader_affine)
test_model(model_no_pooling, test_loader)
print('-'*50)
#結果顯示，透過maxpooling的CNN模型，對於平移的辨識效果較好
