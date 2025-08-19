import torch.nn as nn

class MNIST_EMP_Model(nn.Module):
    """支持11类（0-9 + EMP）的CNN模型"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 11)  # 输出11个类别
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
