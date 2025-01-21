import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ResNet
from utils import load_data, preprocess_data, create_data_loader

# 加载数据
X_train, y_train = load_data('CHD_dataset/csv_dataset/train_data.csv')
X_val, y_val = load_data('CHD_dataset/csv_dataset/val_data.csv')
X_test, y_test = load_data('CHD_dataset/csv_dataset/test_data.csv')

# 数据预处理
X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)

# 创建数据加载器
test_loader = create_data_loader(X_test, y_test, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
model.load_state_dict(torch.load('saved_models/model_epoch_50.pth'))
model.eval()

# 测试模型
with torch.no_grad():
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算每个标签的性能指标
    for i in range(5):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])

        print(f"Label {i}: Accuracy={acc}")

    # 计算整体性能指标
    overall_acc = 0.00000000
    for i in range(5):
        tmp = accuracy_score(all_labels[:, i], all_preds[:, i])
        overall_acc += tmp
    print(f"Overall: Accuracy={overall_acc/5}")