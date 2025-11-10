import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from FFT_PSF import compute_psf, FFT_PSF_for_training, torch_apply_psf
from Aberation_cnn import AberrationCNN, AberrationLoss, PaperParams,AberrationLossV2
from Data_loader import create_synthetic_image, create_training_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from Extract_PSF import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------- 1. 新增：创建最佳模型专属保存文件夹 --------------------------
    # 定义文件夹路径（可自定义，如"best_models"）
    best_model_dir = "best_pre_correction_models"
    # 若文件夹不存在则创建，exist_ok=True避免重复创建报错
    os.makedirs(best_model_dir, exist_ok=True)
    # 用于记录已保存的最佳模型信息：列表元素为 (损失值, 模型文件路径)
    saved_best_models = []
    # 设定保留的最大模型数量
    max_keep_models = 10

    # 2. 定义波长和像差系数
    lambda_values = np.array([486, 587, 656]) * 1e-9
    aberration_coeffs = np.array([
        [0.4548, -0.0365, 2.0154, 0.9962],  # 蓝光
        [0.4202, -0.0506, 1.6590, 0.8220],  # 绿光
        [0.3905, -0.0509, 1.4816, 0.7348]  # 红光
    ])
    # aberration_coeffs = np.array([
    #     [0.4548, -0.0218, 0.7161, 0.3540],  # 蓝光
    #     [0.4202, -0.0302, 0.5895, 0.2921],  # 绿光
    #     [0.3905, -0.0304, 0.5264, 0.2611]  # 红光
    # ])

    # 3. 计算 PSF
    # PSF = compute_psf(lambda_values, aberration_coeffs,g=8.48e-3, visualize=False)
    PSF=extract_psf(file_path="PSF.txt",visual=False)
    
    # 4. 加载训练/验证图片路径
    train_dir = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\"
    val_dir = "F:\\BaiduNetdiskDownload\\COCO_2014\\val2014\\"
    train_image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.endswith('.jpg')]
    val_image_paths = [os.path.join(val_dir, img) for img in os.listdir(val_dir) if img.endswith('.jpg')]

    if not train_image_paths:
        print("No training images found. Using synthetic data.")
        train_image_paths = []
    if not val_image_paths:
        print("No validation images found. Using synthetic data.")
        val_image_paths = []

    # 5. 创建数据集
    # 此处所得到的train_clean_images和val_clean_images都是“干净图像”的Pytorch向量列表
    # 列表中的每个元素是“单张干净图像”的向量(torch.Tensor)，维度为(C,H,W)，数值范围归一化到[0,1]
    train_clean_images, _ = create_training_dataset(PSF, image_paths=train_image_paths,
                                                    synthetic_count=100 if not train_image_paths else 0, image_size=PaperParams.EI_SIZE[0],
                                                    device=device, max_images=5000)
    val_clean_images, _ = create_training_dataset(PSF, image_paths=val_image_paths,
                                                  synthetic_count=50 if not val_image_paths else 0, image_size=PaperParams.EI_SIZE[1],
                                                  device=device, max_images=200)

    # 6. 定义 Dataset 和 DataLoader
    class PreCorrectionDataset(Dataset):
        def __init__(self, clean_images):
            self.clean_images = clean_images
        def __len__(self):
            return len(self.clean_images)
        def __getitem__(self, idx):
            return self.clean_images[idx]

    # 将干净图像张量封装到PreCorrectionDataset中，通过DataLoader按批次输入模型，作为训练/验证的“标签”，用于计算模型预校正效果的损失
    train_loader = DataLoader(PreCorrectionDataset(train_clean_images), batch_size=PaperParams.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PreCorrectionDataset(val_clean_images), batch_size=PaperParams.BATCH_SIZE, shuffle=False)

    # 7. 初始化模型、损失函数、优化器
    model = AberrationCNN().to(device)
    criterion = AberrationLossV2(omega=0.7,bright_weight=2.0).to(device)
    optimizer = optim.SGD(model.parameters(), lr=PaperParams.LR, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3)

    # 8. 训练循环（核心：最佳模型保存与筛选）
    epochs = PaperParams.EPOCHS
    best_val_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        # 训练阶段
        running_loss = 0.0
        for clean_batch in train_loader:
            clean_batch = clean_batch.to(device)
            optimizer.zero_grad()
            pre_corrected = model(clean_batch)
            pre_corrected = torch.clamp(pre_corrected, 0.0, 1.0)
            simulated = torch.zeros_like(pre_corrected)
            for i in range(pre_corrected.size(0)):
                simulated[i] = torch_apply_psf(PSF, pre_corrected[i])
            loss = criterion(simulated, clean_batch,pre_corrected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Average Loss: {avg_train_loss:.6f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                pre_corrected_val = model(val_batch)
                simulated_val = torch.zeros_like(pre_corrected_val)
                for i in range(pre_corrected_val.size(0)):
                    simulated_val[i] = torch_apply_psf(PSF, pre_corrected_val[i])
                val_loss += criterion(simulated_val, val_batch).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Val Average Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)

        # -------------------------- 2. 核心修改：最佳模型保存与筛选 --------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 生成带「时间+损失值」的文件名（便于识别性能）
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 文件名格式：模型名_时间_损失值.pth（损失值保留4位小数，避免过长）
            model_filename = f"best_model_{current_time}_loss{avg_val_loss:.4f}.pth"
            # 拼接完整保存路径（专属文件夹 + 文件名）
            model_save_path = os.path.join(best_model_dir, model_filename)

            # 保存新的最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ 新最佳模型保存：{model_save_path}")

            # 将新模型信息加入列表：(损失值, 保存路径)
            saved_best_models.append( (avg_val_loss, model_save_path) )

            # -------------------------- 3. 筛选：仅保留最好的10个模型 --------------------------
            # 按损失值升序排序（损失越小，模型越好，排在前面）
            saved_best_models.sort(key=lambda x: x[0])
            # 若模型数量超过10个，删除超出的旧模型（从第11个开始）
            if len(saved_best_models) > max_keep_models:
                # 获取需要删除的模型（排序后第10个之后的）
                models_to_delete = saved_best_models[max_keep_models:]
                # 更新列表：只保留前10个
                saved_best_models = saved_best_models[:max_keep_models]

                # 遍历删除超出的模型文件
                for loss_del, path_del in models_to_delete:
                    if os.path.exists(path_del):
                        os.remove(path_del)
                        print(f"🗑️ 删除旧模型（超出10个）：{os.path.basename(path_del)} (损失：{loss_del:.4f})")

    # 9. 保存最终模型（可选：也放入专属文件夹）
    final_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(best_model_dir, f"final_model_{final_time}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n📦 最终模型保存：{final_model_path}")

    # 10. 测试与可视化（保持原逻辑不变）
    model.eval()
    with torch.no_grad():
        test_img_path = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg" if os.path.exists(
            "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg") else None
        if test_img_path:
            test_img = Image.open(test_img_path).convert("RGB").resize((PaperParams.EI_SIZE[0],PaperParams.EI_SIZE[1]))
            test_array = np.array(test_img).astype(np.float32) / 255.0
        else:
            test_array = create_synthetic_image(size=PaperParams.EI_SIZE[1])

        test_tensor = torch.from_numpy(test_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        pre_corrected_test = model(test_tensor)[0]
        pre_corrected_np = pre_corrected_test.detach().cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((pre_corrected_np * 255).astype(np.uint8)).save("pre_corrected_EI.png")

        simulated_test = torch_apply_psf(PSF, pre_corrected_test)
        simulated_np = simulated_test.detach().cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(test_array)
        plt.title("Original EI")
        plt.subplot(1, 3, 2)
        plt.imshow(pre_corrected_np)
        plt.title("Pre-corrected EI")
        plt.subplot(1, 3, 3)
        plt.imshow(simulated_np)
        plt.title("Simulated EI (after PSF)")
        plt.tight_layout()
        plt.show()

    print("测试完成，预校正 EI 已保存为 pre_corrected_EI.png")

    # 11. 计算测试损失（修复原逻辑）
    loss_value, loss1_value = test()
    print(f"预校正后模拟成像 Loss vs 原始 EI: {loss_value:.6f}")
    print(f"原始 EI 直接模拟成像 Loss: {loss1_value:.6f}")


def test():
    # 保持原test函数逻辑不变，仅修复返回值匹配问题
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_img_path = "pre_corrected_EI.png"
    ini_img_path = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg"

    if not os.path.exists(pre_img_path) or not os.path.exists(ini_img_path):
        print("Error: Pre-corrected EI or Original EI not found.")
        return float('inf'), float('inf')

    pre_img = Image.open(pre_img_path).convert("RGB").resize((104, 104))
    pre_array = np.array(pre_img).astype(np.float32) / 255.0
    ini_img = Image.open(ini_img_path).convert("RGB").resize((104, 104))
    ini_array = np.array(ini_img).astype(np.float32) / 255.0

    pre_tensor = torch.from_numpy(pre_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    ini_tensor = torch.from_numpy(ini_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    lambda_values = np.array([486, 587, 656]) * 1e-9
    aberration_coeffs = np.array([
        [0.4548, -0.0218, 0.7161, 0.3540],
        [0.4202, -0.0302, 0.5895, 0.2921],
        [0.3905, -0.0304, 0.5264, 0.2611]
    ])
    PSF = torch.from_numpy(compute_psf(lambda_values, aberration_coeffs, visualize=False)).float().to(device)

    simulated_pre = torch_apply_psf(PSF, pre_tensor)
    if simulated_pre.dim() == 3:
        simulated_pre = simulated_pre.unsqueeze(0)

    K = torch_apply_psf(PSF, ini_tensor)
    if K.dim() == 3:
        K_1 = K.unsqueeze(0)

    criterion = AberrationLoss().to(device)
    loss = criterion(simulated_pre, ini_tensor)
    loss1 = criterion(K_1, ini_tensor)

    return loss.item(), loss1.item()


if __name__ == "__main__":
    main()