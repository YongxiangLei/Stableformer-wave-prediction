import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.enc_in       # 输入维度 (encoder input size)
        self.hidden_size = configs.d_model     # 隐藏层大小 (dimension model)
        self.output_size = configs.c_out       # 输出维度
        self.dropout_rate = configs.dropout     # Dropout 概率
        self.history_length = configs.his_len   # history length
        self.prediction_length = configs.pred_len

        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size * self.history_length, self.hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)  # 输出层
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass through the MLP model.
        Args:
            x_enc (tensor): 输入数据，形状为 [batch_size, seq_len, input_size].
        
        Returns:
            Tensor of shape [batch_size, future_steps, output_size] (多步预测).
        """
        # 展平输入数据
        batch_size = x_enc.size(0)
        x_enc_flat = x_enc.view(batch_size, -1)  # 展平为 [batch_size, input_size * seq_len]

        # 通过第一个全连接层
        x = torch.relu(self.fc1(x_enc_flat))  # 经过激活函数

        # 应用 Dropout
        x = self.dropout(x)

        # 通过输出层
        output = self.fc2(x)  # 形状为 [batch_size, output_size]

        # 为输出添加预测步维度
        output = output.unsqueeze(1).expand(-1, self.prediction_length, -1)  # 形状为 [batch_size, pred_len, output_size]

        return output  # 返回最终输出

    def predict_future(self, x_enc, future_steps):
        """
        未来多步预测。
        Args:
            x_enc (tensor): 输入序列，形状为 [batch_size, seq_len, input_size].
            future_steps (int): 预测未来的步数。
        
        Returns:
            Tensor of shape [batch_size, future_steps, output_size] (未来预测).
        """
        # 展平输入
        batch_size = x_enc.size(0)
        x_enc_flat = x_enc.view(batch_size, -1)

        # 初始预测
        prediction = self.fc2(torch.relu(self.fc1(x_enc_flat))).unsqueeze(1)  # 形状: [batch_size, 1, output_size]
        preds = [prediction]  # 存储预测

        # 递归预测未来步数
        for _ in range(future_steps - 1):
            # 使用最后一个预测结果作为输入进行下一步预测
            prediction = self.fc2(torch.relu(self.fc1(prediction.view(batch_size, -1))))
            prediction = prediction.unsqueeze(1)
            preds.append(prediction)

        # 在时间维度上连接所有预测
        preds = torch.cat(preds, dim=1)  # 形状: [batch_size, future_steps, output_size]

        return preds



# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.input_size = configs.enc_in       # Input dimension
#         self.hidden_size = configs.d_model     # Hidden layer size
#         self.output_size = configs.c_out       # Output dimension
#         self.num_layers = configs.e_layers     # Number of fully connected layers
#         self.dropout_rate = configs.dropout    # Dropout probability

#         # Fully connected layers
#         layers = []
#         layers.append(nn.Linear(self.input_size, self.hidden_size))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(self.dropout_rate))
        
#         for _ in range(self.num_layers - 1):
#             layers.append(nn.Linear(self.hidden_size, self.hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(self.dropout_rate))
        
#         # Final output layer
#         layers.append(nn.Linear(self.hidden_size, self.output_size))
        
#         self.fc = nn.Sequential(*layers)

#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
#         """
#         Forward pass through the fully connected neural network.
#         Args:
#             x_enc (tensor): Encoder input, shape [batch_size, seq_len, input_size].
#             x_dec (tensor): Decoder input, shape [batch_size, pred_len, input_size].
        
#         Returns:
#             Tensor of shape [batch_size, pred_len, output_size].
#         """
#         batch_size, seq_len, input_size = x_enc.size()

#         # Reshape input into 2D tensor [batch_size*seq_len, input_size]
#         x = x_enc.reshape(-1, input_size)

#         # Pass through the fully connected layers
#         output = self.fc(x)

#         # Reshape back to [batch_size, seq_len, output_size]
#         output = output.view(batch_size, seq_len, self.output_size)

#         # Return only the predictions for the decoder sequence
#         return output[:, -x_dec.size(1):, :]

#     def predict_future(self, x_enc, future_steps):
#         """
#         Recursive multi-step future prediction using the last output as input for the next step.
#         Args:
#             x_enc (tensor): Input sequence of shape [batch_size, seq_len, input_size].
#             future_steps (int): Number of steps to predict into the future.
        
#         Returns:
#             Tensor of shape [batch_size, future_steps, output_size].
#         """
#         batch_size, seq_len, input_size = x_enc.size()

#         # Initial prediction based on the last time step
#         prediction = self.fc(x_enc[:, -1, :]).unsqueeze(1)  # Shape: [batch_size, 1, output_size]
#         preds = [prediction]  # Store predictions

#         # Recursive prediction for future steps
#         for _ in range(future_steps - 1):
#             prediction = self.fc(prediction).unsqueeze(1)
#             preds.append(prediction)

#         preds = torch.cat(preds, dim=1)  # Shape: [batch_size, future_steps, output_size]
#         return preds


# # import torch
# # import torch.nn as nn

# # class NN(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
# #         super(NN, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers
        
# #         # 创建全连接层列表
# #         self.layers = nn.ModuleList()
# #         self.layers.append(nn.Linear(input_size, hidden_size))  # 第一层

# #         # 添加隐藏层
# #         for _ in range(1, num_layers):
# #             self.layers.append(nn.Linear(hidden_size, hidden_size))

# #         # 添加输出层
# #         self.layers.append(nn.Linear(hidden_size, output_size))
# #         self.relu = nn.ReLU()

# #     def forward(self, x):
# #         # x 应该是 (batch_size, sequence_length, input_size)
# #         batch_size = x.size(0)
# #         sequence_length = x.size(1)
        
# #         # 将输入形状调整为 (batch_size * sequence_length, input_size)
# #         x = x.view(batch_size * sequence_length, -1)  # 展平为 (batch_size * sequence_length, input_size)
        
# #         for i in range(self.num_layers):
# #             x = self.relu(self.layers[i](x))
        
# #         out = self.layers[-1](x)  # 最后一层是输出层
        
# #         # 恢复为 (batch_size, sequence_length, output_size)
# #         out = out.view(batch_size, sequence_length, -1)
        
# #         # 返回最后一个时间步的输出: (batch_size, output_size)
# #         return out[:, -1, :]  # 选择最后一个时间步的输出

# # if __name__ == "__main__":
# #     # 验证NN模型运行
# #     input_size = 1     # 输入特征维度
# #     hidden_size = 20
# #     num_layers = 2
# #     output_size = 1    
# #     batch_size = 32
# #     sequence_length = 21  # 序列长度

# #     model = NN(input_size, hidden_size, output_size, num_layers)

# #     # 生成随机输入数据
# #     # 形状: (batch_size, sequence_length, input_size)
# #     sample_input = torch.randn(batch_size, sequence_length, input_size)
# #     print("Sample input shape:", sample_input.shape)

# #     # 通过模型进行前向传播
# #     output = model(sample_input)

# #     # 打印输出
# #     print("Output shape:", output.shape)  # 应该是 (batch_size, output_size)
# #     print("Output:", output)



