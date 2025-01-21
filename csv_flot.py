import pandas as pd

# 读取 CSV 文件
file_path = 'CHD_dataset/csv_dataset/val_data.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 指定需要转换为浮点数的列名
columns_to_convert = ['FBG', 'HbA1c', 'TC', 'HDL-C', 'Apo A1', 'Apo B', 'LDL-C', 'TG']

# 将指定列转换为浮点数类型
# for col in columns_to_convert:
#     # 将非数值型数据转换为 NaN
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#     print(col)

# 用均值填充 NaN 值
# df.fillna(df.mean(), inplace=True)

# 确保转换后的列是浮点数类型
df[columns_to_convert] = df[columns_to_convert].astype(float)

# 打印前几行数据以检查转换结果
print(df.head())

# 如果需要，可以将处理后的数据保存回 CSV 文件
df.to_csv('processed_file.csv', index=False)