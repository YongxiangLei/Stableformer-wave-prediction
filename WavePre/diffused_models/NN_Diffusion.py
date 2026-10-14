import torch
import torch.nn as nn

class NN_Diffusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(NN_Diffusion, self).__init__()

        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)  # 输出mu
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 输出sigma

        self.relu = nn.ReLU()
        self.num_layers = num_layers

        # Diffusion parameters
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        # 添加噪声以模拟扩散过程
        noise = torch.randn_like(x) * torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)  # Reshape for broadcasting
        x_noisy = torch.sqrt(alpha_bar_t) * x + noise

        # 前向传播
        out = self.relu(self.fc1(x_noisy))
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
    timesteps = 1000

    model = NN_Diffusion(input_size, hidden_size, output_size, timesteps=timesteps)
    x = torch.randn(batch_size, seq_len, input_size)
    t = torch.randint(0, timesteps, (batch_size,))  # 随机选择时间步
    mu, sigma = model(x, t)
    print("Mu shape:", mu.shape)
    print("Sigma shape:", sigma.shape)
