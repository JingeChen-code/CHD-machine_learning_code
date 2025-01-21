import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

# 设置日志配置
logging.basicConfig(filename='disease_prediction_svm.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

for i in range(5):
    model = SVC(probability=True, kernel='rbf', max_iter=200)
    model.fit(X_train_scaled, y_train.iloc[:, i])
    models.append(model)

    # 记录训练过程中的性能变化
    train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train.iloc[:, i], cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 记录到日志
    for size, train_mean, train_std, test_mean, test_std in zip(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
        logging.info(f'Label {i+1} - Train Size: {size} - Train Score: {train_mean:.4f} (+/- {train_std:.4f}) - Test Score: {test_mean:.4f} (+/- {test_std:.4f})')

    # # 绘制学习曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.title(f'Learning Curve for Label {i+1}')
    # plt.xlabel('Training examples')
    # plt.ylabel('Score')
    # plt.legend(loc="best")
    # plt.savefig(f'learning_curve_label_{i+1}.png')
    # plt.close()

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