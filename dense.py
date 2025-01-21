import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 设置日志配置
logging.basicConfig(filename='disease_prediction_nn.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# 构建神经网络模型
def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 初始化模型列表
models = []

for i in range(5):
    model = build_model(X_train_scaled.shape[1], 1)
    model.fit(X_train_scaled, y_train.iloc[:, i], epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val.iloc[:, i]), verbose=1)
    models.append(model)

# 预测验证集和测试集
y_val_pred = np.column_stack([model.predict(X_val_scaled).round().flatten() for model in models])
y_test_pred = np.column_stack([model.predict(X_test_scaled).round().flatten() for model in models])

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