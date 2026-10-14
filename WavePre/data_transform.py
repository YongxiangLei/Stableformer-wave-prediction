import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque

# 设置数据路径和窗口大小
data_folder = "Cases"
window_size = 21
test_size = 0.2

def load_csv_files(data_folder):
    """
    从指定文件夹加载所有 CSV 文件，并将其内容读取为 numpy 数组。

    参数:
    - data_folder: str, 存放 CSV 文件的文件夹路径。

    返回:
    - all_data: list, 包含每个 CSV 文件的数据的列表。
    """
    all_data = []
    
    # 遍历文件夹中的所有 CSV 文件
    for file in os.listdir(data_folder):
        if file.endswith(".csv") and file.startswith("SS"):
            file_path = os.path.join(data_folder, file)
            # 读取 CSV 文件
            data = pd.read_csv(file_path)
            all_data.append(data.values) 

    return all_data

def normalize_data(data):
    """
    对输入数据进行标准化。

    参数:
    - data: numpy.ndarray, 输入数据。

    返回:
    - scaled_data: numpy.ndarray, 标准化后的数据。
    """
    scaler = StandardScaler()
    data = data.reshape(-1, 1)  # 确保数据为二维数组
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def create_timestamps_ds(series, timestep_size=window_size):
    """
    创建时间序列数据集，将输入序列转换为具有时间戳的特征集和相应标签。

    参数:
    - series: 1D numpy array, 输入序列。
    - timestep_size: int, 滑动窗口大小。

    返回:
    - features: torch.tensor, 特征数据。
    - labels: torch.tensor, 标签数据。
    """
    time_stamps = []
    labels = []
    aux_deque = deque(maxlen=timestep_size)

    # 初始化 deque，填充 0
    for _ in range(timestep_size):
        aux_deque.append(0)

    # 生成时间戳
    for i in range(len(series) - 1):
        aux_deque.append(series[i])
        time_stamps.append(list(aux_deque))

    # 生成标签
    for i in range(len(series) - 1):
        labels.append(series[i + 1])

    assert len(time_stamps) == len(labels), "时间戳和标签长度不一致"

    # 转换为 torch tensor 并返回
    features = torch.tensor(time_stamps[timestep_size:]).float()
    labels = torch.tensor(labels[timestep_size:]).float()

    return features, labels

def split_dataset(features, labels, test_size):
    """
    使用train_test_split划分数据集为训练集和测试集

    参数：
    -features: torch.tensor, 特征数据
    -labels: torch.tensor, 标签数据
    -test_size: float, 测试集占比。
    
    返回：
    -train_features, test_features: 划分后的训练和测试特征
    -train_labels, test_labels: 划分后的训练和测试标签
    """

    #将特征和标签转换为numpy数组以兼容sklearn的train_test_split
    features_np = features.numpy()
    labels_np = labels.numpy()

    train_features, test_features, train_labels, test_labels  = train_test_split(features_np, labels_np, test_size=test_size,shuffle=True)

    #转换为 torch tensor
    train_features = torch.tensor(train_features, dtype=torch.float32)
    test_features = torch.tensor(test_features,dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels,dtype=torch.float32)

    return train_features, test_features, train_labels,test_labels


if __name__ == "__main__":
    # 加载所有 CSV 文件
    all_csv_data = load_csv_files(data_folder)
    
    # 处理每个 CSV 文件的数据
    for i, csv_data in enumerate(all_csv_data):
        # 归一化数据
        normalized_data = normalize_data(csv_data)
        
        # 创建时间序列数据集
        features, labels = create_timestamps_ds(normalized_data.flatten(), timestep_size=window_size)

        # 划分训练集和测试集
        train_features, test_features, train_labels, test_labels = split_dataset(features, labels, test_size=test_size)

        # 2. Reshape the features and labels
        train_features = train_features.unsqueeze(-1)  # Change shape to (num_samples, sequence_length, batch_size), 此处batch_size设置为1！
        test_features = test_features.unsqueeze(-1)    # Change shape to (num_samples, sequence_length, batch_size)

        train_labels = train_labels.unsqueeze(-1)      # Change shape to (num_samples, batch_size)
        test_labels = test_labels.unsqueeze(-1)        # Change shape to (num_samples, batch_size)

        # 输出划分结果的形状
        print(f"文件 SS{i + 1} 的训练集特征 shape: {train_features.shape}, 训练集标签 shape: {train_labels.shape}")
        print(f"文件 SS{i + 1} 的测试集特征 shape: {test_features.shape}, 测试集标签 shape: {test_labels.shape}\n")


                  
