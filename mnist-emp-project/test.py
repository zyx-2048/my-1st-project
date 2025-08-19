import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MNIST_EMP_Model
from utils.custom_dataset import EMPDataset, CombinedDataset
from torchvision.datasets import MNIST
import os
import time

# === 用户需修改的路径变量 ===
EMP_DATA_DIR = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\data\emp"
MODEL_PATH = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\mnist_emp_model.pth"
MNIST_DATA_DIR = r"D:\WAIT-before-to-git\my-1st-project\mnist-emp-project\data"
# ===========================

print("="*50)
print(f"开始测试，使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"EMP数据目录: {EMP_DATA_DIR}")
print(f"模型路径: {MODEL_PATH}")
print(f"MNIST数据目录: {MNIST_DATA_DIR}")
print("="*50)
time.sleep(2)  # 给用户时间查看配置

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载测试数据集
print("加载测试数据集...")
test_mnist = MNIST(root=MNIST_DATA_DIR, train=False, download=True, transform=transform)
test_emp = EMPDataset(root_dir=EMP_DATA_DIR, transform=transform, is_train=False)
combined_test = CombinedDataset(test_mnist, test_emp)
dist = combined_test.get_class_distribution()
print(f"合并后测试集: 总计 {len(combined_test)} 张图片")
print(f"  - MNIST(0-9): {dist['MNIST(0-9)']} 张")
print(f"  - EMP(10): {dist['EMP(10)']} 张")

test_loader = DataLoader(combined_test, batch_size=128, shuffle=False, num_workers=0)  # Windows上num_workers设为0

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_EMP_Model().to(device)

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

# 加载模型
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("模型加载成功")

# 测试准确率
correct = 0
total = 0
class_correct = [0] * 11
class_total = [0] * 11

print("\n开始测试...")
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 统计每个类别的准确率
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

test_time = time.time() - start_time
print(f"测试完成，耗时: {test_time:.1f}秒")

# 打印结果
print("\n" + "="*50)
print(f"整体准确率: {100 * correct / total:.2f}%")
print(f"测试样本总数: {total}")
print("\n各类别准确率:")
for i in range(10):
    if class_total[i] > 0:
        print(f"  数字 {i}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
print(f"  EMP(10): {100 * class_correct[10] / class_total[10]:.2f}% ({class_correct[10]}/{class_total[10]})")
print("="*50)
