import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['FBG', 'HbA1c', 'TC', 'HDL-C', 'Apo A1', 'Apo B', 'LDL-C', 'TG']].values
    y = df[['AMI', 'UA', 'NSTEMI', 'STEMI', 'CHD']].values
    return X, y



def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def create_data_loader(X, y, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)