import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from Aberation_cnn import AberrationCNN,AberrationLoss
from FFT_PSF import compute_psf, torch_apply_psf
import matplotlib.pyplot as plt
import os

'''
//本代码的作用是计算在不同g值下，成像LOSS最小的值并保存。
以那张图片为基准呢
'''
lambda_values=np.array([486,587,656])*1e-9  # 波长，单位米
# lOSS：0.4270
aberration_coeffs1 = np.array([
    [0.4548, -0.0365, 2.0154, 0.9962],  # 蓝光
    [0.4202, -0.0506, 1.6590, 0.8220],  # 绿光
    [0.3905, -0.0509, 1.4816, 0.7348]   # 红光
])

test_image_path="E:\\桌面\\微信图片_20251019131600_122_386.png"
test_img = Image.open(test_image_path).convert("RGB")
test_array = np.array(test_img) / 255.0  # 归一化到 [0, 1]
test_tensor = torch.from_numpy(test_array.transpose(2, 0, 1)).float().unsqueeze(0)
list_g=[]

for i in np.arange(1e-3,30e-3,1e-4):
    g=i
    PSF= compute_psf(lambda_values, aberration_coeffs1,g=g,visualize=False)


    simulated_test = torch_apply_psf(PSF, test_tensor)
    if simulated_test.dim() == 3:  # 如果丢失批量维度，添加回去
        simulated_test = simulated_test.unsqueeze(0)
    elif simulated_test.dim() != 4:  # 确保是 4 维张量
        raise ValueError(f"Unexpected dimension of simulated_test: {simulated_test.shape}")

    criterion = AberrationLoss()
    loss = criterion(simulated_test, test_tensor)
    list_g.append([g,loss.item()])
    print(f"g={g:.4f} m时，LOSS={loss.item():.4f}")

