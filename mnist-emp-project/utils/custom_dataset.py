import os
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

class EMPDataset(Dataset):
    """自定义EMP符号数据集"""
    def __init__(self, root_dir, transform=None, is_train=True):
        # 确保路径使用正确的分隔符
        root_dir = os.path.normpath(root_dir)
        sub_dir = "train-1600" if is_train else "test-400"
        
        # 构建完整路径
        self.root_dir = os.path.join(root_dir, sub_dir)
        
        # 检查路径是否存在
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"目录不存在: {self.root_dir}")
        
        self.transform = transform
        
        # 获取所有PNG文件
        self.image_files = [f for f in os.listdir(self.root_dir) 
                           if f.lower().endswith(".png")]
        
        # 检查是否有文件
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"在 {self.root_dir} 中未找到PNG文件")
        
        print(f"成功加载 {len(self.image_files)} 张EMP图片，路径: {self.root_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # 转为灰度图
        label = 10  # EMP标签设为10
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CombinedDataset(Dataset):
    """合并MNIST和EMP数据集"""
    def __init__(self, mnist_dataset, emp_dataset):
        self.dataset = ConcatDataset([mnist_dataset, emp_dataset])
        self.mnist_len = len(mnist_dataset)
        self.emp_len = len(emp_dataset)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_class_distribution(self):
        return {"MNIST(0-9)": self.mnist_len, "EMP(10)": self.emp_len}
