import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from FFT_PSF import compute_psf, FFT_PSF_for_training, torch_apply_psf
from Aberation_cnn import AberrationCNN, AberrationLoss, PaperParams
from Data_loader import create_synthetic_image, create_training_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 定义波长和像差系数（论文示例参数）
    lambda_values = np.array([486, 587, 656]) * 1e-9
    aberration_coeffs = np.array([
        [0.4548, -0.0218, 0.7161, 0.3540],  # 蓝光
        [0.4202, -0.0302, 0.5895, 0.2921],  # 绿光
        [0.3905, -0.0304, 0.5264, 0.2611]  # 红光
    ])

    # 2. 计算 PSF
    PSF = compute_psf(lambda_values, aberration_coeffs, visualize=False)

    # 3. 动态加载训练集图片
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

    # 4. 创建训练和验证数据集
    train_clean_images, _ = create_training_dataset(PSF, image_paths=train_image_paths,
                                                    synthetic_count=100 if not train_image_paths else 0, image_size=104,
                                                    device=device,max_images=1000)
    val_clean_images, _ = create_training_dataset(PSF, image_paths=val_image_paths,
                                                  synthetic_count=50 if not val_image_paths else 0, image_size=104,
                                                  device=device,max_images=200)

    # 5. 定义自定义 Dataset
    class PreCorrectionDataset(Dataset):
        def __init__(self, clean_images):
            self.clean_images = clean_images

        def __len__(self):
            return len(self.clean_images)

        def __getitem__(self, idx):
            return self.clean_images[idx]

    train_dataset = PreCorrectionDataset(train_clean_images)
    val_dataset = PreCorrectionDataset(val_clean_images)

    # 6. 定义 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=PaperParams.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=PaperParams.BATCH_SIZE, shuffle=False)

    # 7. 定义预校正 CNN 模型、损失函数和优化器
    model = AberrationCNN().to(device)
    criterion = AberrationLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=PaperParams.LR, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3)

    # 8. 训练循环：最小化 Simulated EI 与原始 EI 的差异
    epochs = PaperParams.EPOCHS
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for clean_batch in train_loader:
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()

            # CNN 输出预校正 EI
            pre_corrected = model(clean_batch)

            # 应用 PSF 得到 Simulated EI
            simulated = torch.zeros_like(pre_corrected)
            for i in range(pre_corrected.size(0)):
                simulated[i] = torch_apply_psf(PSF, pre_corrected[i])

            # 计算损失
            loss = criterion(simulated, clean_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Average Loss: {avg_train_loss:.6f}")

        # 验证
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_pre_correction_model.pth")

    # 9. 保存最终模型
    torch.save(model.state_dict(), "pre_correction_model.pth")
    print("模型已保存为 pre_correction_model.pth")

    # 10. 测试：使用模型校正测试图像，并验证成像效果
    model.eval()
    with torch.no_grad():
        test_img_path = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg" if os.path.exists(
            "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg") else None
        if test_img_path:
            test_img = Image.open(test_img_path).convert("RGB").resize((104, 104))
            test_array = np.array(test_img).astype(np.float32) / 255.0
        else:
            test_array = create_synthetic_image(size=104)

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

    # 11. 计算预校正 EI 的成像结果与原始 EI 的 LOSS
    loss_value = test()
    print(f"Pre-corrected EI Simulated Loss vs Original EI: {loss_value:.6f}")


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预校正 EI 和原始 EI
    pre_img_path = "pre_corrected_EI.png"
    ini_img_path = "F:\\BaiduNetdiskDownload\\COCO_2014\\train2014\\COCO_train2014_000000000025.jpg"

    if not os.path.exists(pre_img_path) or not os.path.exists(ini_img_path):
        print("Error: Pre-corrected EI or Original EI not found.")
        return float('inf')

    pre_img = Image.open(pre_img_path).convert("RGB").resize((104, 104))
    pre_array = np.array(pre_img).astype(np.float32) / 255.0
    ini_img = Image.open(ini_img_path).convert("RGB").resize((104, 104))
    ini_array = np.array(ini_img).astype(np.float32) / 255.0

    # 转换为张量，保留批量维度
    pre_tensor = torch.from_numpy(pre_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    ini_tensor = torch.from_numpy(ini_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # 计算 PSF 并转换为张量
    lambda_values = np.array([486, 587, 656]) * 1e-9
    aberration_coeffs = np.array([
        [0.4548, -0.0218, 0.7161, 0.3540],
        [0.4202, -0.0302, 0.5895, 0.2921],
        [0.3905, -0.0304, 0.5264, 0.2611]
    ])
    PSF = torch.from_numpy(compute_psf(lambda_values, aberration_coeffs, visualize=False)).float().to(device)

    # 应用 PSF 得到模拟成像结果
    simulated_pre = torch_apply_psf(PSF, pre_tensor)
    if simulated_pre.dim() == 3:  # 如果丢失批量维度，添加回去
        simulated_pre = simulated_pre.unsqueeze(0)

    K=torch_apply_psf(PSF,ini_tensor)
    if K.dim() == 3:  # 如果丢失批量维度，添加回去
        K_1 = K.unsqueeze(0)
    # 计算损失（使用与训练相同的损失函数）
    criterion = AberrationLoss().to(device)
    loss = criterion(simulated_pre, ini_tensor)
    loss1=criterion(K_1, ini_tensor)

    return loss.item(), loss1.item()

if __name__ == "__main__":
    # main()
    print(test())


