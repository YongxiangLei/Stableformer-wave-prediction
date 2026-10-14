import torch
import torch.nn as nn

class LSTM_Diffusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(LSTM_Diffusion, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_sigma = nn.Linear(hidden_size, output_size)
        
        # Diffusion parameters
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        noise = torch.randn_like(x) * torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        x_noisy = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1) * x + noise

        out, _ = self.lstm(x_noisy, (h0, c0))
        pred_mu = self.fc_mu(out[:, -1, :])
        pred_sigma = torch.exp(self.fc_sigma(out[:, -1, :]))
        return pred_mu, pred_sigma



