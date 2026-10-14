import torch
import torch.nn as nn

class NN_Prob(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(NN_Prob, self).__init__()

        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)  # 输出mu
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 输出sigma

        self.relu = nn.ReLU()
        self.num_layers = num_layers

    def forward(self, x):
        # 前向传播
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        
        mu = self.fc_mu(out)  # 生成均值
        sigma = self.fc_sigma(out)  # 生成标准差
        
        # 确保sigma为正数
        sigma = torch.exp(sigma)
        
        # 只返回最后一个时间步的 mu 和 sigma
        mu = mu[:, -1, :]
        sigma = sigma[:, -1, :]
        
        return mu, sigma

# 测试
if __name__ == "__main__":
    batch_size, seq_len, input_size = 32, 21, 1
    hidden_size = 64
    output_size = 1

    model = NN_Prob(input_size, hidden_size, output_size)
    x = torch.randn(batch_size, seq_len, input_size)
    mu, sigma = model(x)
    print("Mu shape:", mu.shape)
    print("Sigma shape:", sigma.shape)
