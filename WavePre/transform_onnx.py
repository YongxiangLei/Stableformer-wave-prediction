# import torch
# import os
# import onnx
# import onnxruntime as ort
# from base_models import RNN, LSTMModel, TCN, StableFormer, Transformer, GRU, MLP

# # 配置参数类
# class Configs:
#     enc_in = 1           # 输入大小
#     dec_in = 1           # 解码器输入大小
#     c_out = 1            # 输出大小
#     hidden_size = 128    # 隐藏层大小
#     d_model = 64         # 模型维度
#     embed = 'timeF'      # 嵌入类型
#     freq = 'h'           # 频率
#     dropout = 0.1        # Dropout率
#     kernel_size = 3      # 卷积核大小
#     factor = 5           # 注意力因子
#     n_heads = 8          # 注意力头数量
#     d_ff = 128           # 前馈层维度
#     e_layers = 2         # 编码器层数
#     d_layers = 1         # 解码器层数
#     activation = 'gelu'  # 激活函数
#     learning_rate = 0.0001
#     his_len = 60         # 历史长度
#     pred_len = 40        # 预测长度（多步）

# configs = Configs()

# # 加载模型函数，根据文件名判断模型类型
# def load_model(model_path, configs):
#     model_name = os.path.basename(model_path).split('_')[0]
    
#     if model_name == 'RNN':
#         model = RNN.Model(configs)
#     elif model_name == 'TCN':
#         model = TCN.Model(configs)
#     elif model_name == 'LSTMModel':
#         model = LSTMModel.Model(configs)
#     elif model_name == 'Transformer':
#         model = Transformer.Model(configs)
#     elif model_name == 'GRU':
#         model = GRU.Model(configs)
#     elif model_name == 'StableFormer':
#         model = StableFormer.Model(configs)
#     elif model_name == 'MLP':
#         model = MLP.Model(configs)
#     else:
#         raise ValueError(f"Unknown model type: {model_name}")

#     model.load_state_dict(torch.load(model_path))
#     return model

# # 转换 pth 文件为 onnx 文件
# def convert_pth_to_onnx(pth_file, onnx_file, configs):
#     # 加载模型
#     model = load_model(pth_file, configs)
#     model.eval()  # 设置模型为评估模式

#     # 选择设备
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
    
#     # 创建随机输入
#     dummy_enc_input = torch.randn(1, configs.his_len, configs.enc_in).to(device)  # Encoder input shape
#     dummy_dec_input = torch.randn(1, configs.pred_len, configs.dec_in).to(device)  # Decoder input shape

#     # 检查形状
#     print("Encoder Input Shape:", dummy_enc_input.shape, "Decoder Input Shape:", dummy_dec_input.shape)

#     # ONNX export
#     torch.onnx.export(
#         model,
#         (dummy_enc_input, None, dummy_dec_input, None),  # Pass both encoder and decoder inputs
#         onnx_file,
#         input_names=['encoder_input', 'encoder_mark', 'decoder_input', 'decoder_mark'],
#         output_names=['output'],
#         dynamic_axes={
#             'encoder_input': {0: 'batch_size', 1: 'seq_len_enc'},
#             'decoder_input': {0: 'batch_size', 1: 'seq_len_dec'},
#             'output': {0: 'batch_size', 1: 'pred_len'}
#         },
#         opset_version=12
#     )
    
#     # 检查并验证 ONNX 模型
#     onnx_model = onnx.load(onnx_file)
#     try:
#         onnx.checker.check_model(onnx_model)
#         print(f"{onnx_file} 模型成功转换并通过检查!")
#     except onnx.checker.ValidationError as e:
#         print(f"{onnx_file} 转换失败: {e}")


# # 主函数，遍历模型文件夹并转换所有模型
# def main():
#     models_folder = './models'
#     onnx_folder = './onnx'
#     os.makedirs(onnx_folder, exist_ok=True)

#     for filename in os.listdir(models_folder):
#         if filename.endswith('short.pth'):
#             pth_file = os.path.join(models_folder, filename)
#             onnx_file = os.path.join(onnx_folder, filename.replace('.pth', '.onnx'))

#             print(f"Converting {pth_file} to {onnx_file}...")
#             convert_pth_to_onnx(pth_file, onnx_file, configs)
#             print(f"Saved {onnx_file}")

# if __name__ == "__main__":
#     main()


# import os
# import torch
# import onnx
# import numpy as np
# import onnxruntime as ort  # 导入 onnxruntime 以进行预测
# from base_models import LSTMModel  # 导入你的神经网络模型

# # 设备选择
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device:', device)

# # 配置参数类
# class Configs:
#     enc_in = 1           # 输入大小
#     dec_in = 1           # 解码器输入大小
#     c_out = 1            # 输出大小
#     hidden_size = 128    # LSTM的隐藏层大小
#     d_model = 64         # 模型维度
#     embed = 'timeF'      # 嵌入类型
#     freq = 'h'           # 频率
#     dropout = 0.1        # Dropout率
#     kernel_size = 3      # 卷积核大小
#     factor = 5           # 注意力因子
#     n_heads = 8          # 注意力头数量
#     d_ff = 128           # 前馈层维度
#     e_layers = 2         # 编码器层数
#     d_layers = 1         # 解码器层数
#     activation = 'gelu'  # 激活函数
#     learning_rate = 0.0001
#     his_len = 60         # 历史长度
#     pred_len = 40        # 预测长度（多步）

# configs = Configs()

# # 检查ONNX文件夹是否存在，否则创建
# onnx_dir = './onnx'
# os.makedirs(onnx_dir, exist_ok=True)

# # 循环转换SS1到SS6模型
# for i in range(1, 7):
#     model_name = f'SS{i}'  # 动态生成模型名称
#     model_path = f'./models/LSTMModel_{model_name}_short.pth'
#     onnx_path = os.path.join(onnx_dir, f'LSTMModel_{model_name}_short.onnx')
    
#     # 加载模型
#     model = LSTMModel.Model(configs)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.eval().to(device)
    
#     # 创建随机输入
#     dummy_enc_input = torch.randn(1, configs.his_len, configs.enc_in).to(device)  # Encoder input shape
#     dummy_dec_input = torch.randn(1, configs.pred_len, configs.dec_in).to(device)  # Decoder input shape

#     # 检查形状
#     print("Encoder Input Shape:", dummy_enc_input.shape, "Decoder Input Shape:", dummy_dec_input.shape)

#     # ONNX export
#     torch.onnx.export(
#         model,
#         (dummy_enc_input, None, dummy_dec_input, None),  # Pass both encoder and decoder inputs
#         onnx_path,
#         input_names=['encoder_input', 'encoder_mark', 'decoder_input', 'decoder_mark'],
#         output_names=['output'],
#         dynamic_axes={
#             'encoder_input': {0: 'batch_size', 1: 'seq_len_enc'},
#             'decoder_input': {0: 'batch_size', 1: 'seq_len_dec'},
#             'output': {0: 'batch_size', 1: 'pred_len'}
#         },
#         opset_version=14
#     )

#     # 检查ONNX模型
#     onnx_model = onnx.load(onnx_path)
#     try:
#         onnx.checker.check_model(onnx_model)
#         print(f'{onnx_path} 模型成功转换并通过检查!')
#     except onnx.checker.ValidationError as e:
#         print(f'{onnx_path} 转换失败: {e}')

#     # 打印计算图
#     print(onnx.helper.printable_graph(onnx_model.graph))
    
#     # 使用 onnxruntime 运行模型预测
#     ort_session = ort.InferenceSession(onnx_path)

#     # 准备输入数据
#     # 需要将 decoder_input 设为 dummy_dec_input
#     input_data = {
#         'encoder_input': dummy_enc_input.cpu().numpy(),
#         'decoder_input': dummy_dec_input.cpu().numpy(),
#     }

#     # 运行预测
#     ort_outs = ort_session.run(['output'], input_data)

#     # 输出预测结果
#     print(f'{onnx_path} 模型的预测输出:\n', ort_outs[0])


import os
import torch
import onnx
import numpy as np
import onnxruntime as ort  # 导入 onnxruntime 以进行预测
from base_models import LSTMModel  # 导入你的神经网络模型

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 配置参数类
class Configs:
    enc_in = 1           # Input size
    dec_in = 1           # Decoder input size
    c_out = 1            # Output size
    hidden_size = 128    # Hidden size for MLP
    d_model = 64         # Model dimension 64
    embed = 'timeF'      # Embedding type
    freq = 'h'           # Frequency
    dropout = 0.1        # Dropout rate
    kernel_size = 3      # Kernel size
    factor = 5           # Attention factor
    n_heads = 8          # Number of attention heads
    d_ff = 128           # Feedforward layer dimension
    e_layers = 2         # Number of encoder layers
    d_layers = 1         # Number of decoder layers
    activation = 'gelu'  # Activation function
    learning_rate = 0.0001
    his_len = 60        # History length
    pred_len = 40       # Prediction length (multi-step)

configs = Configs()

# 检查ONNX文件夹是否存在，否则创建
onnx_dir = './onnx'
os.makedirs(onnx_dir, exist_ok=True)

# 循环转换SS1到SS6模型
for i in range(1, 7):
    model_name = f'SS{i}'  # 动态生成模型名称
    model_path = f'./models/LSTMModel_{model_name}_short.pth'
    onnx_path = os.path.join(onnx_dir, f'LSTMModel_{model_name}_short.onnx')
    
    # 加载模型
    model = LSTMModel.Model(configs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval().to(device)
    
    # 创建随机输入
    dummy_enc_input = torch.randn(1, configs.his_len, configs.enc_in).to(device)  # Encoder input shape

    # 检查形状
    print("Encoder Input Shape:", dummy_enc_input.shape)

    # ONNX export with only encoder_input
    torch.onnx.export(
        model,
        (dummy_enc_input,),  # Pass only encoder input
        onnx_path,
        input_names=['encoder_input'],
        output_names=['output'],
        dynamic_axes={
            'encoder_input': {0: 'batch_size', 1: 'seq_len_enc'},
            'output': {0: 'batch_size', 1: 'pred_len'}
        },
        opset_version=14
    )

    # 检查ONNX模型
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print(f'{onnx_path} 模型成功转换并通过检查!')
    except onnx.checker.ValidationError as e:
        print(f'{onnx_path} 转换失败: {e}')

    # 打印计算图
    print(onnx.helper.printable_graph(onnx_model.graph))
    
    # 使用 onnxruntime 运行模型预测
    ort_session = ort.InferenceSession(onnx_path)

    # 准备输入数据
    input_data = {
        'encoder_input': dummy_enc_input.cpu().numpy(),
    }

    # 运行预测
    ort_outs = ort_session.run(['output'], input_data)

    # 输出预测结果
    print(f'{onnx_path} 模型的预测输出:\n', ort_outs[0])
