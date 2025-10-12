import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei（黑体），兼容英文
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示异常

# PSF计算函数（从FFT_PSF.py简化实现）
def compute_psf(lambda_values, aberration_coefficients,
                g=29.9e-3, r_max=0.491e-3,
                num_out=101, x_range=0.1e-3, y_range=0.1e-3,
                Nr=30, Ntheta=30, H=0):
    x_out, y_out = np.meshgrid(
        np.linspace(-x_range, x_range, num_out),
        np.linspace(-y_range, y_range, num_out)
    )
    r = np.linspace(0, r_max, Nr)
    theta = np.linspace(0, 2 * np.pi, Ntheta)
    rr, tt = np.meshgrid(r, theta)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    rho_norm = rr / r_max
    PSF_total = np.zeros((num_out, num_out))
    for idx_lambda, lam in enumerate(lambda_values):
        k = 2 * np.pi / lam
        W040, W131, W222, W220 = aberration_coefficients[idx_lambda]
        W = (W040 * rho_norm ** 4 +
             W131 * rho_norm ** 3 * np.cos(tt) +
             W222 * (H ** 2) * rho_norm ** 2 * np.cos(tt) ** 2 +
             W220 * (H ** 2) * rho_norm ** 2) * lam
        PSF_lambda = np.zeros((num_out, num_out))
        for i in range(num_out):
            for j in range(num_out):
                x0 = x_out[i, j]
                y0 = y_out[i, j]
                cos_term = (x0 / g) * np.cos(tt) + (y0 / g) * np.sin(tt)
                integrand = np.exp(-1j * 2 * np.pi * W / lam) * \
                            np.exp(-1j * k * rr * cos_term) * rr
                val = np.sum(integrand) * dr * dtheta
                PSF_lambda[i, j] = abs(val) ** 2 / (lam ** 2 * g ** 2)
        PSF_total += PSF_lambda
    PSF_normalized = PSF_total / np.max(PSF_total)
    return PSF_normalized

# torch_apply_psf 函数
def torch_apply_psf(psf, image):
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 添加批次维度 -> [1, C, H, W]

    B, C, H, W = image.shape

    if not isinstance(psf, torch.Tensor):
        psf = torch.from_numpy(psf).float()

    psf = psf.unsqueeze(0).unsqueeze(0)  # [1, 1, H_psf, W_psf]

    device = image.device
    psf = psf.to(device)

    Y_sim_list = []
    for c in range(C):
        Y = F.conv2d(image[:, c:c+1, :, :], psf, padding=psf.shape[-1]//2, groups=1)
        Y_sim_list.append(Y)
    Y_sim = torch.cat(Y_sim_list, dim=1)

    Y_sim = torch.clamp(Y_sim, 0, float('inf'))
    min_val = Y_sim.view(B, C, -1).min(dim=2)[0].unsqueeze(2).unsqueeze(3)
    max_val = Y_sim.view(B, C, -1).max(dim=2)[0].unsqueeze(2).unsqueeze(3)
    mask = max_val > min_val
    Y_sim = torch.where(mask, (Y_sim - min_val) / (max_val - min_val + 1e-8), Y_sim)

    return Y_sim.squeeze(0) if B == 1 else Y_sim

# 合成图像函数（作为备用）
def create_synthetic_image(size=104):
    img = np.zeros((size, size, 3), dtype=np.float32)
    img_type = np.random.randint(0, 4)
    if img_type == 0:
        block_size = np.random.randint(16, 64)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    img[i:i + block_size, j:j + block_size] = np.random.rand(3)
    elif img_type == 1:
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            center_x = np.random.randint(size // 4, 3 * size // 4)
            center_y = np.random.randint(size // 4, 3 * size // 4)
            radius = np.random.randint(size // 16, size // 8)
            color = np.random.rand(3)
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2
            img[mask] = color
    elif img_type == 2:
        for i in range(size):
            for j in range(size):
                img[i, j] = [i / size, j / size, (i + j) / (2 * size)]
    else:
        img = np.random.rand(size, size, 3).astype(np.float32)
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], sigma=2.0)
    return img

# 测试函数
def test_torch_apply_psf_with_real_image(img_path="pre_corrected_EI.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载真实图片或使用合成备用
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((104, 104))
        real_image_np = np.array(img).astype(np.float32) / 255.0
        print("使用本地真实图片进行验证")
    else:
        real_image_np = create_synthetic_image(size=104)
        print("本地图片不存在，使用合成图片作为备用")

    real_image = torch.from_numpy(real_image_np.transpose(2, 0, 1)).float().to(device)  # [3, 104, 104]
    real_image.requires_grad_(True)

    # 生成论文中的PSF
    lambda_values = np.array([486, 587, 656]) * 1e-9
    aberration_coeffs = np.array([
        [0.4548, -0.0218, 0.7161, 0.3540],
        [0.4202, -0.0302, 0.5895, 0.2921],
        [0.3905, -0.0304, 0.5264, 0.2611]
    ])
    psf_np = compute_psf(lambda_values, aberration_coeffs)
    psf = torch.from_numpy(psf_np).float().to(device)

    # 应用PSF卷积
    aberrated = torch_apply_psf(psf, real_image)

    # 验证输出
    print("=== 验证结果 ===")
    print(f"输入图像形状: {real_image.shape}")
    print(f"输出图像形状: {aberrated.shape}")
    print(f"输出范围: [{aberrated.min().item():.4f}, {aberrated.max().item():.4f}]")

    # 梯度验证
    target = torch.rand_like(aberrated).to(device)
    loss = F.mse_loss(aberrated, target)
    loss.backward()
    print(f"梯度存在: {real_image.grad is not None}")

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(real_image_np)
    plt.title("原始真实图像")

    plt.subplot(1, 2, 2)
    plt.imshow(aberrated.detach().cpu().numpy().transpose(1, 2, 0))
    plt.title("带像差图像")
    plt.tight_layout()
    plt.show()

# 运行测试（替换为您的图片路径）
test_torch_apply_psf_with_real_image(img_path="F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000009.jpg")

