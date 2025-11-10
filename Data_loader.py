import matplotlib.image as mpimg
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from FFT_PSF import FFT_PSF_for_training
from Aberation_cnn import *


class PSFDataset(Dataset):
    '''
    通过加载真实图像，对其进行预处理后，使用点扩散函数（PSF）进行卷积操作，生成对应的 “像差图像”，最终返回 “干净图像” 与 “像差图像” 的张量对，用于模型的输入（像差图像）和标签（干净图像）。
    '''
    def __init__(self, image_paths, PSF, image_size=104, device="cpu"):
        self.image_paths = image_paths
        self.PSF = PSF
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()  # 转换为Pytorch张量，(C,H,W)，自动归一化到[0,1]
        ])

    def __len__(self):
        return len(self.image_paths) #必须实现的方法，返回数据集的样本数量（即图像路径列表的长度）

    def apply_psf(self, image_array):
        """对图像做PSF卷积，模拟像差"""
        if image_array.ndim == 3:# 若为多通道图像
            Y_sim = np.zeros_like(image_array)
            for c in range(image_array.shape[0]):  # (C,H,W)，逐个通道的进行卷积
                Y_sim[c] = fftconvolve(image_array[c], self.PSF, mode="same")
        else:# 处理单通道图像
            Y_sim = fftconvolve(image_array, self.PSF, mode="same")
        # 归一化
        Y_sim = np.clip(Y_sim, 0, None)
        if Y_sim.max() > Y_sim.min():
            Y_sim = (Y_sim - Y_sim.min()) / (Y_sim.max() - Y_sim.min())
        return Y_sim

    def __getitem__(self, idx):
        '''
        使用Dataloader加载数据时，会通过索引调用该方法，生成单组训练样本（clean_img和aberrated_img）
        该函数是重写Dataset中的函数
        :param idx:
        :return:
        '''
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        clean_tensor = self.transform(img)  # (C,H,W)

        # 转 numpy 后卷积
        clean_np = clean_tensor.numpy()
        aberrated_np = self.apply_psf(clean_np)

        aberrated_tensor = torch.from_numpy(aberrated_np).float()

        return clean_tensor.to(self.device), aberrated_tensor.to(self.device)


def create_synthetic_image(size=256):
    """创建合成训练图像"""
    '''
    创建多种类型的合成训练图像
    '''
    img = np.zeros((size, size, 3), dtype=np.float32)

    # 随机选择图像类型
    img_type = np.random.randint(0, 4)

    if img_type == 0:
        # 棋盘图案
        block_size = np.random.randint(16, 64)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    img[i:i + block_size, j:j + block_size] = np.random.rand(3)

    elif img_type == 1:
        # 随机圆形
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
        # 渐变图案
        for i in range(size):
            for j in range(size):
                img[i, j] = [i / size, j / size, (i + j) / (2 * size)]

    else:
        # 随机噪声图案
        img = np.random.rand(size, size, 3).astype(np.float32)
        # 应用低通滤波器使其更平滑
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], sigma=2.0)

    return img


def batch_generator(clean_images, aberrated_images, batch_size=16, shuffle=True):
    """
    批量数据生成器，支持数据打乱和分批处理

    参数:
        clean_images: 清洁图像列表
        aberrated_images: 像差图像列表
        batch_size: 批量大小
        shuffle: 是否打乱数据

    生成:
        (clean_batch, aberrated_batch): 批量数据
    """

    indices = list(range(len(clean_images)))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        clean_batch = torch.stack([clean_images[idx] for idx in batch_indices])
        aberrated_batch = torch.stack([aberrated_images[idx] for idx in batch_indices])
        yield clean_batch, aberrated_batch



def create_training_dataset(PSF, image_paths=None, synthetic_count=0, image_size=256, device='cpu', max_images=1000,convolution=False):
    """
    创建训练数据集

    参数:
        PSF: 点扩散函数
        image_paths: 图像路径列表（可选）
        synthetic_count: 合成图像数量
        image_size: 图像尺寸
        device: torch设备
        max_images: 最大图像数量限制（默认 1000）

    返回:
        clean_images: 原始图像tensor列表
        aberrated_images: 像差图像tensor列表
    """
    '''
    批量创建训练数据集，支持真实图像和合成图像，限制总图像数
    '''
    clean_images = []
    aberrated_images = []

    # 处理真实图像，限制为 max_images
    if image_paths:
        # 取前 max_images 张图片
        image_paths = image_paths[:max_images]
        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    # 加载并预处理图像
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img).astype(np.float32) / 255.0

                    # 应用PSF卷积
                    clean_tensor, aberrated_tensor = FFT_PSF_for_training(
                        PSF, img_array, return_torch=True, device=device,convolution=False
                    )

                    clean_images.append(clean_tensor)
                    aberrated_images.append(aberrated_tensor)
                    if len(clean_images) >= max_images:  # 达到上限后停止
                        break

                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")

    # 创建合成图像，仅在真实图像不足时补充
    remaining_count = max(0, max_images - len(clean_images))
    for i in range(min(remaining_count, synthetic_count)):
        synthetic_img = create_synthetic_image(image_size)
        clean_tensor, aberrated_tensor = FFT_PSF_for_training(
            PSF, synthetic_img, return_torch=True, device=device,convolution=False
        )
        clean_images.append(clean_tensor)
        aberrated_images.append(aberrated_tensor)
    
     # 校验数据格式
    for i, (clean, aberrated) in enumerate(zip(clean_images, aberrated_images)):
        if not isinstance(clean, torch.Tensor) or not isinstance(aberrated, torch.Tensor):
            raise TypeError(f"第{i}个样本不是张量，类型：{type(clean)}, {type(aberrated)}")


    return clean_images, aberrated_images