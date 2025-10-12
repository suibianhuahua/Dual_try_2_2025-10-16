import matplotlib.image as mpimg
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn.functional as F

def compute_psf(lambda_values, aberration_coefficients,
                g=29.9e-3, r_max=0.491e-3,
                num_out=101, x_range=0.1e-3, y_range=0.1e-3,
                Nr=30, Ntheta=30, H=0,
                visualize=True):
    """
    计算带像差的点扩散函数 (PSF)

    参数
    ----------
    lambda_values : list or np.ndarray
        波长数组 (m)，例如 [486e-9, 587e-9, 656e-9]
    aberration_coefficients : np.ndarray
        每个波长对应的像差系数，形状 (len(lambda_values), 4)，
        顺序为 [W040(球差), W131(彗差), W222(像散), W220(场曲)]
    g : float
        微透镜到显示平面的距离 (m)
    r_max : float
        微透镜半径 (m)
    num_out : int
        输出平面网格点数
    x_range, y_range : float
        输出平面坐标范围 (m)
    Nr, Ntheta : int
        瞳孔平面离散采样点数
    H : float
        归一化视场坐标
    visualize : bool
        是否绘制 2D 和 3D 图像

    返回
    ----------
    PSF_normalized : np.ndarray
        归一化后的 PSF (num_out x num_out)
    """

    # 成像平面网格
    x_out, y_out = np.meshgrid(
        np.linspace(-x_range, x_range, num_out),
        np.linspace(-y_range, y_range, num_out)
    )

    # 积分网格 (瞳孔平面)
    r = np.linspace(0, r_max, Nr)
    theta = np.linspace(0, 2 * np.pi, Ntheta)
    rr, tt = np.meshgrid(r, theta)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    rho_norm = rr / r_max

    # PSF 累加 (三波长合成)
    PSF_total = np.zeros((num_out, num_out))
    for idx_lambda, lam in enumerate(lambda_values):
        k = 2 * np.pi / lam
        W040, W131, W222, W220 = aberration_coefficients[idx_lambda]

        # 波前像差
        W = (W040 * rho_norm ** 4 +
             W131 * rho_norm ** 3 * np.cos(tt) +
             W222 * (H ** 2) * rho_norm ** 2 * np.cos(tt) ** 2 +
             W220 * (H ** 2) * rho_norm ** 2) * lam

        # 输出平面逐点计算
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

    # 归一化
    PSF_normalized = PSF_total / np.max(PSF_total)

    # 可视化
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.imshow(PSF_normalized, extent=[-x_range*1e3, x_range*1e3, -y_range*1e3, y_range*1e3],
                   cmap='gray', aspect='equal')
        plt.colorbar(label='Normalized Intensity')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('PSF Normalized Intensity Distribution')
        plt.tight_layout()
        plt.show()

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_out*1e3, y_out*1e3, PSF_normalized,
                               cmap='jet', edgecolor='none')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Normalized Intensity')
        ax.set_title('PSF 3D Surface Plot')
        plt.colorbar(surf)
        plt.tight_layout()
        plt.show()

    return PSF_normalized

def FFT_PSF_for_training(PSF, image_array, return_torch=True, device='cpu'):
    """
    对图像应用PSF卷积，专门用于神经网络训练
    返回torch tensor格式，维度为(C,H,W)

    参数:
        PSF: 点扩散函数 (numpy数组)
        image_array: 输入图像数组，shape为 (H, W) 或 (H, W, C)
        return_torch: 是否返回torch.Tensor格式
        device: torch设备类型

    返回:
        如果 return_torch=True:
            返回 (原始图像tensor, 卷积后图像tensor)，范围 [0, 1]，shape为 (C, H, W)
        否则:
            返回 (原始图像array, 卷积后图像array)，范围 [0, 1]
    """
    # 确保输入是float32格式，范围[0,1]
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)

    if image_array.max() > 1.0:
        image_array = image_array / 255.0

    # 处理不同维度的图像
    if image_array.ndim == 2:
        # 灰度图：添加通道维度
        image_array = image_array[..., np.newaxis]
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        # RGBA：去除alpha通道
        image_array = image_array[..., :3]
    # 对每个通道进行卷积
    if image_array.ndim == 3:
        Y_sim = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            Y_sim[..., c] = fftconvolve(image_array[..., c], PSF, mode="same")
    else:
        Y_sim = fftconvolve(image_array, PSF, mode="same")
        if Y_sim.ndim == 2:
            Y_sim = Y_sim[..., np.newaxis]

    # 归一化到 [0,1]
    Y_sim = np.clip(Y_sim, 0, None)  # 确保非负
    Y_sim_max = Y_sim.max()
    Y_sim_min = Y_sim.min()
    if Y_sim_max > Y_sim_min:
        Y_sim = (Y_sim - Y_sim_min) / (Y_sim_max - Y_sim_min)

    if return_torch:
        # 转换为torch tensor，并调整维度顺序 (H, W, C) -> (C, H, W)
        clean_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().to(device)
        aberrated_tensor = torch.from_numpy(Y_sim.transpose(2, 0, 1)).float().to(device)
        return clean_tensor, aberrated_tensor
    else:
        return image_array, Y_sim


def FFT_PSF(PSF,Pic_path):
    """
    对图像应用PSF卷积并可视化结果

    参数:
        PSF: 点扩散函数 (numpy数组)
        Pic_path: 输入图像路径
        save_path: 保存卷积结果的路径（默认为 "convolved_result.png"）
        visualize: 是否显示对比图
        return_arrays: 是否返回numpy数组（用于训练）

    返回:
        如果 return_arrays=True:
            返回 (原始图像, 卷积后图像) 两个numpy数组，范围 [0, 1]
        否则:
            返回 None
    """
    # 读取EI图片 (替换为你自己的路径)
    EI = plt.imread(Pic_path)

    # 如果是 RGBA 图片，去掉 alpha 通道
    if EI.ndim == 3 and EI.shape[2] == 4:
        EI = EI[:, :, :3]

    # 转换到 [0,1]
    EI = EI.astype(np.float32)
    if EI.max() > 1:
        EI = EI / 255.0


    # 灰度图情况
    if EI.ndim == 2:
        Y_sim = fftconvolve(EI, PSF, mode="same")

    # 彩色图情况 (RGB 每个通道卷积一次)
    elif EI.ndim == 3:
        Y_sim = np.zeros_like(EI)
        for c in range(3):
            Y_sim[..., c] = fftconvolve(EI[..., c], PSF, mode="same")

    # 归一化到 [0,1] 方便显示
    Y_sim = (Y_sim - Y_sim.min()) / (Y_sim.max() - Y_sim.min())

    Y_sim_save=(Y_sim*255).astype(np.uint8)

    mpimg.imsave("convolved_result.png",Y_sim_save)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(EI, cmap='gray')
    plt.title("Original EI")

    plt.subplot(1, 2, 2)
    plt.imshow(Y_sim, cmap='gray')
    plt.title("EI after PSF convolution")

    plt.tight_layout()
    plt.show()

def torch_apply_psf(psf, image):
    """
    Torch版本的PSF卷积，模拟像差，支持梯度传播
    参数:
        psf: 点扩散函数 (torch tensor, shape [H_psf, W_psf])
        image: 输入图像tensor，shape为 [C, H, W] 或 [B, C, H, W]
    返回:
        aberrated_tensor: 卷积后图像tensor，范围 [0, 1]
    """
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