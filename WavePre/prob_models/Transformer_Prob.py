import torch
import torch.nn as nn
import torch.distributions as dist

class Transformer_Prob(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=2, num_layers=2):
        super(Transformer_Prob, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers
        )
        self.fc_mu = nn.Linear(hidden_size, output_size)  # 均值
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # 标准差

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        mu = self.fc_mu(x[:, -1, :])
        sigma = torch.exp(self.fc_sigma(x[:, -1, :]))  # 确保标准差为正
        return mu, sigma
