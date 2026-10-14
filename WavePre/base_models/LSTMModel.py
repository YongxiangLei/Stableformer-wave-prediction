import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.enc_in       # 输入维度 (encoder input size)
        self.hidden_size = configs.d_model     # 隐藏层大小 (dimension model)
        self.output_size = configs.c_out       # 输出维度
        self.num_layers = configs.e_layers     # LSTM层数
        self.dropout_rate = configs.dropout    # dropout概率
        self.pred_len = configs.pred_len

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout_rate)

        # Fully connected layer for prediction
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass through the LSTM model.
        Args:
            x_enc (tensor): Encoder input, shape [batch_size, seq_len, input_size].
            x_dec (tensor): Decoder input, shape [batch_size, pred_len, input_size].
        
        Returns:
            Tensor of shape [batch_size, pred_len, output_size] (multi-step prediction).
        """
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x_enc)  # Output shape: [batch_size, seq_len, hidden_size]

        # Fully connected layer for output prediction at each time step
        output = self.fc(lstm_out)  # Shape: [batch_size, seq_len, output_size]

        # Return only the predictions for the decoder sequence
        return output[:, -x_dec.size(1):, :]
        #return output[:, -self.config.pred_len:, :]

    def predict_future(self, x_enc, future_steps):
        """
        Recursive multi-step future prediction using the last output as input for the next step.
        Args:
            x_enc (tensor): Input sequence of shape [batch_size, seq_len, input_size].
            future_steps (int): Number of steps to predict into the future.
        
        Returns:
            Tensor of shape [batch_size, future_steps, output_size] (future predictions).
        """
        # Forward pass through the LSTM to get the initial predictions
        lstm_out, (h_n, c_n) = self.lstm(x_enc)  # Output shape: [batch_size, seq_len, hidden_size]
        last_hidden_state = lstm_out[:, -1, :]  # Take the last hidden state

        # Initial prediction based on the last hidden state
        prediction = self.fc(last_hidden_state).unsqueeze(1)  # Shape: [batch_size, 1, output_size]
        preds = [prediction]  # Store predictions

        # Recursive prediction for future steps
        for _ in range(future_steps - 1):
            # Use the last prediction as input for the next step
            lstm_out, (h_n, c_n) = self.lstm(prediction, (h_n, c_n))
            prediction = self.fc(lstm_out[:, -1, :]).unsqueeze(1)
            preds.append(prediction)

        # Concatenate all predictions along the time dimension
        preds = torch.cat(preds, dim=1)  # Shape: [batch_size, future_steps, output_size]

        return preds





# import torch
# import torch.nn as nn

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)

#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # Output from the last time step

#         return out

# if __name__ == "__main__":
#     # 验证LSTM模型运行
#     input_size = 1     #输入特征维度： 一维wave excitation moment 
#     hidden_size = 20
#     num_layer = 2
#     output_size = 1    
#     batch_size = 32
#     sequence_length = 21    # 序列长度

#     model = LSTM(input_size,hidden_size,num_layer,output_size)

#     # Generate random input data
#     # Shape: (batch_size, sequence_length, input_size)
#     sample_input = torch.randn(batch_size, sequence_length, input_size)
#     print(sample_input.shape)

#     # Forward pass through the model
#     output = model(sample_input)

#     # Print the output
#     print("Output shape:", output.shape)  # Should be (batch_size, output_size)
#     print("Output:", output) 