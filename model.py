import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=False)
        # self.resnet = models.resnet50(weights=False)
         # 修改第一个卷积层以接受单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 5)  # 修改最后的全连接层以匹配输出维度

    def forward(self, x):
        x = x.view(x.size(0), 1, 8, 1)  # 调整输入维度以匹配ResNet的输入
        return self.resnet(x)