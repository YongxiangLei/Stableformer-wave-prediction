import torch
import torch.nn as nn
import torch.distributions as dist

class LSTM_Prob(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Prob, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, output_size)  # 均值
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 标准差

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        mu = self.fc_mu(out[:, -1, :])
        sigma = torch.exp(self.fc_sigma(out[:, -1, :]))  # 确保标准差为正
        return mu, sigma
