import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler

# 设置日志配置
logging.basicConfig(filename='disease_prediction_LR.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取数据
train_data = pd.read_csv('CHD_dataset/csv_dataset/train_data.csv')
val_data = pd.read_csv('CHD_dataset/csv_dataset/val_data.csv')
test_data = pd.read_csv('CHD_dataset/csv_dataset/test_data.csv')

# 特征和标签分离
X_train = train_data[['FBG', 'HbA1c', 'TC', 'HDL-C', 'Apo A1', 'Apo B', 'LDL-C', 'TG']]
y_train = train_data[['AMI', 'UA', 'NSTEMI', 'STEMI', 'CHD']]

X_val = val_data[['FBG', 'HbA1c', 'TC', 'HDL-C', 'Apo A1', 'Apo B', 'LDL-C', 'TG']]
y_val = val_data[['AMI', 'UA', 'NSTEMI', 'STEMI', 'CHD']]

X_test = test_data[['FBG', 'HbA1c', 'TC', 'HDL-C', 'Apo A1', 'Apo B', 'LDL-C', 'TG']]
y_test = test_data[['AMI', 'UA', 'NSTEMI', 'STEMI', 'CHD']]

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 初始化模型列表
models = []
losses = []

for i in range(5):
    model = LogisticRegression(max_iter=1, warm_start=True, solver='saga', penalty='l1')
    losses.append([])  # 初始化损失列表
    for j in range(50):
        model.fit(X_train_scaled, y_train.iloc[:, i])
        y_train_pred_proba = model.predict_proba(X_train_scaled)
        loss = log_loss(y_train.iloc[:, i], y_train_pred_proba)
        losses[i].append(loss)
        logging.info(f'Label {i+1} - Iteration {j+1} - Loss: {loss:.6f}')
    models.append(model)


# 预测验证集和测试集
y_val_pred = np.array([model.predict(X_val_scaled) for model in models]).T
y_test_pred = np.array([model.predict(X_test_scaled) for model in models]).T

# 计算各项指标
def calculate_metrics(y_true, y_pred, label):
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f'{label} - Accuracy: {accuracy:.8f}')
    return accuracy

# 验证集指标
val_metrics = []
for i in range(5):
    metrics = calculate_metrics(y_val.iloc[:, i], y_val_pred[:, i], f'Validation - Label {i+1}')
    val_metrics.append(metrics)

val_metrics_array = np.array(val_metrics)
val_avg_metrics = np.mean(val_metrics_array, axis=0)
logging.info(f'Validation - Average Accuracy: {val_avg_metrics:.8f}')

# 测试集指标
test_metrics = []
for i in range(5):
    metrics = calculate_metrics(y_test.iloc[:, i], y_test_pred[:, i], f'Test - Label {i+1}')
    test_metrics.append(metrics)

test_metrics_array = np.array(test_metrics)
test_avg_metrics = np.mean(test_metrics_array, axis=0)
logging.info(f'Test - Average Accuracy: {test_avg_metrics:.8f}')

