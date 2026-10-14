import torch
import torch.nn as nn
import torch.distributions as dist

class TCN_Prob(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=2):
        super(TCN_Prob, self).__init__()
        self.convs = nn.ModuleList()
        self.layers = num_layers
        self.hidden_size = hidden_size
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.convs.append(nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 
                                        padding=(kernel_size - 1) * dilation, dilation=dilation))
        
        self.fc_mu = nn.Linear(hidden_size, output_size)  # 均值
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 标准差

    def forward(self, x):
        x = x.transpose(1, 2)  # 转换为 (batch_size, input_size, sequence_length)
        
        for conv in self.convs:
            x = torch.relu(conv(x))

        x = x[:, :, -1]  # 取最后的卷积结果
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))  # 确保标准差为正
        return mu, sigma

