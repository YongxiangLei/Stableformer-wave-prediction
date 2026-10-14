import torch
import torch.nn as nn
import torch.distributions as dist

class TCN_Diffusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=2, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(TCN_Diffusion, self).__init__()
        self.convs = nn.ModuleList()
        self.layers = num_layers
        self.hidden_size = hidden_size

        # 构建卷积层
        for i in range(num_layers):
            dilation = 2 ** i
            self.convs.append(nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 
                                        padding=(kernel_size - 1) * dilation, dilation=dilation))

        self.fc_mu = nn.Linear(hidden_size, output_size)  # 输出均值
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 输出标准差

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

        x_noisy = x_noisy.transpose(1, 2)  # 转换为 (batch_size, input_size, sequence_length)
        
        for conv in self.convs:
            x_noisy = torch.relu(conv(x_noisy))

        x_noisy = x_noisy[:, :, -1]  # 取最后的卷积结果
        mu = self.fc_mu(x_noisy)
        sigma = torch.exp(self.fc_sigma(x_noisy))  # 确保标准差为正
        
        return mu, sigma

# 测试
if __name__ == "__main__":
    batch_size, seq_len, input_size = 32, 21, 1
    hidden_size = 64
    output_size = 1
    timesteps = 1000

    model = TCN_Diffusion(input_size, hidden_size, output_size, timesteps=timesteps)
    x = torch.randn(batch_size, seq_len, input_size)
    t = torch.randint(0, timesteps, (batch_size,))  # 随机选择时间步
    mu, sigma = model(x, t)
    print("Mu shape:", mu.shape)
    print("Sigma shape:", sigma.shape)
