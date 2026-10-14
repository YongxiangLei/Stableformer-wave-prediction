import os
import torch
import onnx
import onnxruntime as ort  # 导入 onnxruntime 以进行预测
from base_models import MLP  # 导入你的神经网络模型

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 配置参数类
class Configs:
    enc_in = 1           # 输入大小
    dec_in = 1           # 解码器输入大小
    c_out = 1            # 输出大小
    hidden_size = 128    # MLP的隐藏层大小
    d_model = 64         # 模型维度
    embed = 'timeF'      # 嵌入类型
    freq = 'h'           # 频率
    dropout = 0.1        # Dropout率
    kernel_size = 3      # 卷积核大小
    factor = 5           # 注意力因子
    n_heads = 8          # 注意力头数量
    d_ff = 128           # 前馈层维度
    e_layers = 2         # 编码器层数
    d_layers = 1         # 解码器层数
    activation = 'gelu'  # 激活函数
    learning_rate = 0.0001
    his_len = 600        # 历史长度
    pred_len = 400      # 预测长度（多步）

configs = Configs()

# 检查ONNX文件夹是否存在，否则创建
onnx_dir = './onnx'
os.makedirs(onnx_dir, exist_ok=True)

# 循环转换SS1到SS6模型
for i in range(1, 7):
    model_name = f'SS{i}'  # 动态生成模型名称
    model_path = f'./models/MLP_{model_name}_long.pth'
    onnx_path = os.path.join(onnx_dir, f'MLP_{model_name}_long.onnx')
    
    # 加载模型
    model = MLP.Model(configs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval().to(device)
    
    # 创建随机输入
    dummy_input = torch.randn(1, configs.his_len, 1).to(device)
    print(dummy_input.shape)
    
    # 转换为ONNX格式并保存
    with torch.no_grad():
        torch.onnx.export(
            model,                           # 要转换的模型
            dummy_input,                     # 模型的任意一组输入
            onnx_path,                       # 导出的 ONNX 文件名
            opset_version=11,                # ONNX 算子集版本
            input_names=['input'],           # 输入 Tensor 的名称
            output_names=['output']          # 输出 Tensor 的名称
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
    input_data = dummy_input.cpu().numpy()
    
    # 运行预测
    ort_inputs = {'input': input_data}
    ort_outs = ort_session.run(['output'], ort_inputs)
    
    # 输出预测结果
    print(f'{onnx_path} 模型的预测输出:\n', ort_outs[0])



