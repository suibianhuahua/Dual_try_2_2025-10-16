'''
本代码的作用是利用训练好的模型去验证效果
'''



import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from Aberation_cnn import AberrationCNN
from FFT_PSF import compute_psf, torch_apply_psf
import matplotlib.pyplot as plt
import os

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型结构
model = AberrationCNN().to(device)

# 3. 加载训练好的模型参数
model_path = "pre_correction_model.pth"  # 或 "best_pre_correction_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式
    print(f"成功加载模型: {model_path}")
else:
    print(f"错误: 模型文件 {model_path} 不存在")
    exit()

# 4. 准备输入数据（示例：加载测试图像）
test_img_path = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000036.jpg"
if os.path.exists(test_img_path):
    test_img = Image.open(test_img_path).convert("RGB").resize((104, 104))
    test_array = np.array(test_img).astype(np.float32) / 255.0
else:
    print(f"错误: 测试图像 {test_img_path} 不存在")
    exit()

test_tensor = torch.from_numpy(test_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

# 5. 进行推理
with torch.no_grad():  # 禁用梯度计算
    pre_corrected_test = model(test_tensor)[0]
    pre_corrected_np = pre_corrected_test.detach().cpu().numpy().transpose(1, 2, 0)

# 6. 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_array)
plt.title("Original EI")
plt.subplot(1, 2, 2)
plt.imshow(pre_corrected_np)
plt.title("Pre-corrected EI")
plt.tight_layout()
plt.show()

# 7. （可选）应用 PSF 模拟成像并计算损失
lambda_values = np.array([486, 587, 656]) * 1e-9
aberration_coeffs = np.array([
    [0.4548, -0.0218, 0.7161, 0.3540],
    [0.4202, -0.0302, 0.5895, 0.2921],
    [0.3905, -0.0304, 0.5264, 0.2611]
])
PSF = torch.from_numpy(compute_psf(lambda_values, aberration_coeffs, visualize=False)).float().to(device)
simulated_test = torch_apply_psf(PSF, pre_corrected_test)
# 修正维度不匹配
if simulated_test.dim() == 3:  # 如果丢失批量维度，添加回去
    simulated_test = simulated_test.unsqueeze(0)
elif simulated_test.dim() != 4:  # 确保是 4 维张量
    raise ValueError(f"Unexpected dimension of simulated_test: {simulated_test.shape}")

# 动态调整 transpose 操作
if simulated_test.dim() == 4:  # [B, C, H, W]
    simulated_np = simulated_test.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)  # 移除批次维度后转置
else:
    raise ValueError(f"simulated_test dimension {simulated_test.dim()} not supported for transpose")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(pre_corrected_np)
plt.title("Pre-corrected EI")
plt.subplot(1, 2, 2)
plt.imshow(simulated_np)
plt.title("Simulated EI (after PSF)")
plt.tight_layout()
plt.show()

# 8. （可选）计算与原始 EI 的损失
from Aberation_cnn import AberrationLoss
criterion = AberrationLoss().to(device)
ini_tensor = torch.from_numpy(test_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
loss = criterion(simulated_test, ini_tensor)

simulated_test_1=torch_apply_psf(PSF,test_tensor).unsqueeze(0)
loss1=criterion(simulated_test_1, ini_tensor)


print(f"Simulated Loss vs Original EI: {loss.item():.6f}")
print(f"Initial Loss vs Original EI: {loss1.item():.6f}")

# 9. 保存预校正 EI
Image.fromarray((pre_corrected_np * 255).astype(np.uint8)).save("inferred_pre_corrected_EI.png")
print("预校正 EI 已保存为 inferred_pre_corrected_EI.png")