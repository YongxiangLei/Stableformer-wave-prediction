import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, 
                               padding=(kernel_size - 1) * dilation, 
                               dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.dropout(self.activation(self.conv(x)))

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.enc_in       # 输入维度 (encoder input size)
        self.output_size = configs.c_out       # 输出维度
        self.num_layers = configs.e_layers      # TCN层数
        self.kernel_size = configs.kernel_size  # 卷积核大小
        self.dropout_rate = configs.dropout      # dropout概率
        self.dilations = [2 ** i for i in range(self.num_layers)]

        # TCN层
        self.tcn_layers = nn.ModuleList(
            [TCNBlock(self.input_size if i == 0 else configs.d_model,
                       configs.d_model, self.kernel_size, 
                       self.dilations[i], self.dropout_rate) 
             for i in range(self.num_layers)]
        )

        # Fully connected layer for prediction
        self.fc = nn.Linear(configs.d_model, self.output_size)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass through the TCN model.
        Args:
            x_enc (tensor): Encoder input, shape [batch_size, seq_len, input_size].
            x_dec (tensor): Decoder input, shape [batch_size, pred_len, input_size].
        
        Returns:
            Tensor of shape [batch_size, pred_len, output_size] (multi-step prediction).
        """
        # 转换输入形状为 [batch_size, input_size, seq_len] 以适应1D卷积
        x_enc = x_enc.transpose(1, 2)  # 形状变为 [batch_size, input_size, seq_len]
        
        for layer in self.tcn_layers:
            x_enc = layer(x_enc)  # 通过所有TCN层

        # 转换回 [batch_size, seq_len, output_size] 形状
        output = self.fc(x_enc.transpose(1, 2))  # [batch_size, seq_len, output_size]

        # 返回仅包含预测序列的输出
        return output[:, -x_dec.size(1):, :]

    def predict_future(self, x_enc, future_steps):
        """
        Recursive multi-step future prediction using the last output as input for the next step.
        Args:
            x_enc (tensor): Input sequence of shape [batch_size, seq_len, input_size].
            future_steps (int): Number of steps to predict into the future.
        
        Returns:
            Tensor of shape [batch_size, future_steps, output_size] (future predictions).
        """
        # 转换输入形状为 [batch_size, input_size, seq_len]
        x_enc = x_enc.transpose(1, 2)  # 变形为 [batch_size, input_size, seq_len]
        
        # Forward pass through TCN to get the initial predictions
        for layer in self.tcn_layers:
            x_enc = layer(x_enc)

        last_hidden_state = x_enc[:, :, -1].unsqueeze(-1)  # 取最后的隐藏状态，形状为 [batch_size, d_model, 1]

        # Initial prediction based on the last hidden state
        prediction = self.fc(last_hidden_state.permute(0, 2, 1))  # 变形为 [batch_size, 1, output_size]
        preds = [prediction]  # 存储预测

        # Recursive prediction for future steps
        for _ in range(future_steps - 1):
            # 使用最后的预测作为下一个步骤的输入
            x_enc = torch.cat((x_enc[:, :, 1:], prediction.permute(0, 2, 1)), dim=2)  # 移除最旧的输入并添加最新预测
            for layer in self.tcn_layers:
                x_enc = layer(x_enc)

            last_hidden_state = x_enc[:, :, -1].unsqueeze(-1)
            prediction = self.fc(last_hidden_state.permute(0, 2, 1)).unsqueeze(1)  # 变形为 [batch_size, 1, output_size]
            preds.append(prediction)

        # 将所有预测沿时间维度连接
        preds = torch.cat(preds, dim=1)  # 形状为 [batch_size, future_steps, output_size]

        return preds



# import torch
# import torch.nn as nn

# class TCN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=2):
#         super(TCN, self).__init__()
#         self.convs = nn.ModuleList()
#         self.layers = num_layers
#         self.hidden_size = hidden_size
        
#         # 定义多层卷积
#         for i in range(num_layers):
#             dilation = 2 ** i
#             self.convs.append(nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 
#                                         padding=(kernel_size - 1) * dilation, dilation=dilation))
        
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: (batch_size, sequence_length, input_size)
#         x = x.transpose(1, 2)  # 转换为 (batch_size, input_size, sequence_length)
        
#         # 通过每一层卷积
#         for conv in self.convs:
#             x = torch.relu(conv(x))

#         # 转换为 (batch_size, hidden_size)
#         x = x[:, :, -1]  # 取最后的卷积结果
#         x = self.fc(x)
#         return x


# if __name__ == "__main__":
#     # 验证TCN模型运行
#     input_size = 1        # 输入特征维度： 一维 wave excitation moment 
#     hidden_size = 20
#     output_size = 1    
#     num_layers = 2        # 设置层数
#     batch_size = 32
#     sequence_length = 21   # 序列长度

#     model = TCN(input_size, hidden_size, output_size, num_layers=num_layers)

#     # 生成随机输入数据
#     # Shape: (batch_size, sequence_length, input_size)
#     sample_input = torch.randn(batch_size, sequence_length, input_size)
#     print("Input shape:", sample_input.shape)

#     # 前向传播
#     output = model(sample_input)

#     # 打印输出
#     print("Output shape:", output.shape)  # 应该是 (batch_size, output_size)
#     print("Output:", output)
