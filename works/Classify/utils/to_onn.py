"""
华为云modelarts
将pytorch模型转换成onnx模型
"""
from models.MyResNet.ResNet import CreateResNet50
import torch.onnx

# 加载 PyTorch 模型
model = CreateResNet50(5)
model.load_state_dict(torch.load('../weights/best.pth',map_location=torch.device('cpu')))

# 设置模型输入，包括：通道数，分辨率等
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

# 转换为ONNX模型
torch.onnx.export(model, dummy_input, "../weights/models.onnx", export_params=True)