import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MNIST_EMP_Model
from utils.custom_dataset import EMPDataset, CombinedDataset
from torchvision.datasets import MNIST
import os
import time

# === 用户需修改的路径变量 ===
# 使用原始字符串和绝对路径确保兼容性
EMP_DATA_DIR = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\data\emp"
SAVE_MODEL_PATH = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\mnist_emp_model.pth"
MNIST_DATA_DIR = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\data"
# ===========================

print("="*50)
print(f"开始训练，使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"EMP数据目录: {EMP_DATA_DIR}")
print(f"模型保存路径: {SAVE_MODEL_PATH}")
print(f"MNIST数据目录: {MNIST_DATA_DIR}")
print("="*50)
time.sleep(2)  # 给两秒钟时间查看配置

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 确保MNIST目录存在
os.makedirs(MNIST_DATA_DIR, exist_ok=True)

# 加载MNIST数据集
print("加载MNIST数据集...")
train_mnist = MNIST(root=MNIST_DATA_DIR, train=True, download=True, transform=transform)
print(f"加载MNIST训练集: {len(train_mnist)} 张图片")

# 加载EMP数据集
print("加载EMP数据集...")
train_emp = EMPDataset(root_dir=EMP_DATA_DIR, transform=transform, is_train=True)
print(f"加载EMP训练集: {len(train_emp)} 张图片")

# 合并数据集
print("合并数据集...")
combined_train = CombinedDataset(train_mnist, train_emp)
dist = combined_train.get_class_distribution()
print(f"合并后训练集: 总计 {len(combined_train)} 张图片")
print(f"  - MNIST(0-9): {dist['MNIST(0-9)']} 张")
print(f"  - EMP(10): {dist['EMP(10)']} 张")

train_loader = DataLoader(combined_train, batch_size=128, shuffle=True, num_workers=0)  # Windows上num_workers设为0

# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_EMP_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
model.train()
print("\n开始训练...")
for epoch in range(15):
    epoch_start = time.time()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每100批次打印一次进度
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/15], Batch [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/15] 完成, "
          f"Loss: {total_loss/len(train_loader):.4f}, "
          f"Acc: {100 * correct / total:.2f}%, "
          f"耗时: {epoch_time:.1f}秒")

# 保存模型
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 15
}, SAVE_MODEL_PATH)
print(f"\n模型已保存到: {SAVE_MODEL_PATH}")
