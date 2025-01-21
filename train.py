import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import ResNet
from utils import load_data, preprocess_data, create_data_loader
from sklearn.metrics import accuracy_score

# 指定模型保存的文件夹路径
model_save_dir = 'saved_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 如果文件夹不存在，则创建它
# 加载数据
X_train, y_train = load_data('CHD_dataset/csv_dataset/train_data.csv')
X_val, y_val = load_data('CHD_dataset/csv_dataset/val_data.csv')
X_test, y_test = load_data('CHD_dataset/csv_dataset/test_data.csv')



# 数据预处理
X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)

# 创建数据加载器
train_loader = create_data_loader(X_train, y_train, shuffle=True)
val_loader = create_data_loader(X_val, y_val, shuffle=False)

# 构建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
writer = SummaryWriter('runs/resnet_experiment')
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累加损失
        running_loss += loss.item()
        
        # 打印训练进度
        if (i+1) % 10 == 0:  # 每10个批次打印一次进度
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]')
    
    # 每轮结束后计算验证集上的精度
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 打印每个标签的性能指标
    for i in range(5):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])

        print(f"Label {i}: Accuracy={acc}")

    # 计算整体性能指标
    overall_acc = 0.00000000
    for i in range(5):
        tmp = accuracy_score(all_labels[:, i], all_preds[:, i])
        overall_acc += tmp
    print(f"Overall: Accuracy={overall_acc/5}")

    # 保存模型权重到指定文件夹
    model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
 