import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pytorch_msssim import ssim

'''
2.2 小节的代码复现需要全部放置在该文件中

--------------------------------------------
论文2.2小节的复现内容：
第一步：单微透镜像差校正（基于特征图像损失）
    输入：初始 EI（未校正）、原始 EI（理想无像差）、模拟 EI（有像差，由 2.1 节方法生成）；
    特征提取：通过轻量化 CNN 分别提取 “原始 EI” 和 “模拟 EI” 的结构特征图；
    损失计算：用公式（11）计算两者特征图的强度分布误差（第一损失函数）；
    反向传播：通过 SGD 迭代更新初始 EI 的像素值，最小化第一损失函数，得到 “初步预校正 EI”（校正单微透镜像差）。

第二步：整个 MLA 像差校正（基于重建图像损失）
    光学重建：将 “原始 EI” 和 “模拟 EI” 分别通过 MLA 进行光学重建，得到 “原始重建图像” 和 “模拟重建图像”；
    损失计算：用公式（11）的变体（第二损失函数）计算两种重建图像的强度分布差异（聚焦整个 MLA 的像差，而非单个微透镜）；
    二次反向传播：基于第二损失函数再次更新 “初步预校正 EI”，确保其通过 MLA 重建后，与 “原始重建图像” 一致，最终得到 “全局最优预校正 EI”。

收敛条件：
    设置 “可接受误差阈值”（通过实验验证确定），当 SGD 迭代中 “总损失函数Loss(abe)” 低于阈值且不再下降时，认为预校正 EI 已收敛到全局最优，停止迭代。

'''
class PaperParams:
    """论文中与nn.MSELoss相关的核心参数"""
    EI_SIZE = (104, 104)  # EI分辨率（论文86节：输入图像resize为104×104）
    IN_CHANNELS = 3  # 图像通道数（RGB）
    OMEGA = 0.8  # 公式11中SSIM损失的权重（论文2.2节验证ω=0.8时优化效果最佳）
    DATA_RANGE = 1.0  # 图像像素值范围（归一化到0~1，符合论文批量训练逻辑）
    MSE_REDUCTION = "mean"  # MSE损失归约方式（批量平均，适配论文批量训练）
    LR = 5e-3  # 初始学习率（论文87节）
    BATCH_SIZE = 16  # 批次大小（论文87节）
    EPOCHS = 120  # 训练轮次（论文87节：360轮后学习率下降）

# 3层轻量的卷积神经网络的架构定义
class AberrationCNN(nn.Module):
    def __init__(self,in_ch=3,feat_ch=128):
        super(AberrationCNN, self).__init__()
        #卷积层1：3×3卷积核，保持尺寸
        self.conv1=nn.Conv2d(in_channels=in_ch,out_channels=feat_ch,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(feat_ch)

        #卷积层2：细化特征，聚焦像差相关模式
        self.conv2=nn.Conv2d(in_channels=feat_ch,out_channels=feat_ch,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(feat_ch)

        #逆卷积层：复原到原图通道
        self.deconv=nn.ConvTranspose2d(in_channels=feat_ch,out_channels=in_ch,kernel_size=3,stride=1,padding=1)
        #激活
        self.act=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.act(self.bn1(self.conv1(x)))
        x=self.act(self.bn2(self.conv2(x)))
        x=self.deconv(x)
        #输出为预校正图像
        out=torch.clamp(x,0.0,1)
        return out

    def get_features(self, x):
        """提取conv2后的特征图作为结构特征（用于损失计算）"""
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x  # 返回特征图（非输出图像）

# 论文中所示的公式11，即组合的损失函数定义
# def aberration_loss_func(
#         pred:torch.Tensor,
#         target:torch.Tensor,
#         omega:float=0.8,
#         data_range:float=1.0,
# )->torch.Tensor:
#     """
#     公式11损失函数的函数式实现（像差预校正损失）
#     Args:
#         pred: 预测图像（预校正EI或MLA重建图像），维度[B, C, H, W]
#         target: 目标图像（原始无像差EI或原始重建图像），维度[B, C, H, W]
#         omega: SSIM损失的权重（论文2.2节验证ω=0.8时优化效果最佳）
#         data_range: 图像像素值动态范围（论文中图像归一化到0~1，故默认1.0）
#     Returns:
#         total_loss: 加权融合后的总损失（公式11的计算结果）
#     """
#     Loss_ssim=1-ssim(pred, target,data_range=data_range,size_average=True)
#     Loss_mse=ssim(pred,target,data_range=data_range,size_average=True)
#     Loss_abe=omega*Loss_ssim+(1-omega)*Loss_mse
#
#     return Loss_abe

# 论文中所示的公式11，即组合的损失函数定义
# 使用nn.mes而非F.mse，更加适合端到端的训练
class AberrationLoss(nn.Module):
    def __init__(self):
        super(AberrationLoss, self).__init__()
        # 1. 实例化nn.MSELoss模块（固定归约方式，论文公式12）
        self.mse_loss = nn.MSELoss(reduction=PaperParams.MSE_REDUCTION)
        # 2. 论文公式11的固定参数
        self.omega = PaperParams.OMEGA
        self.data_range = PaperParams.DATA_RANGE

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算像差预校正总损失（论文公式11）
        Args:
            pred: 预测图像（预校正EI或MLA重建图像），维度[B, C, H, W]
            target: 目标图像（原始无像差EI或原始重建图像），维度[B, C, H, W]
        Returns:
            total_loss: 公式11的总损失值
        """
        # 调整维度顺序，确保输入为4维张量
        if pred.dim()!=4:
            pred=pred.unsqueeze(0)
        if target.dim()!=4:
            target=target.unsqueeze(0)



        # 步骤1：计算SSIM损失（论文公式13）：1-SSIM（SSIM越接近1，损失越小）
        loss_ssim = 1 - ssim(
            pred, target,
            data_range=self.data_range,
            size_average=True  # 批量内平均，与MSE损失归约方式一致
        )

        # 步骤2：通过nn.MSELoss计算MSE损失（论文公式12）
        loss_mse = self.mse_loss(pred, target)  # 无需重复设置reduction，实例化时已固定

        # 步骤3：公式11：加权融合两种损失
        total_loss = self.omega * loss_ssim + (1 - self.omega) * loss_mse

        return total_loss

class AberrationLossV2(nn.Module):
    def __init__(self, omega=0.8, brightness_weight=1.0):
        super().__init__()
        self.omega = omega
        self.bw = brightness_weight
        self.mse = nn.MSELoss()
        self.ssim_mod = ssim

    def forward(self, pred_recon, target, pre_corrected):
        # 1. 主损失
        loss_ssim = 1 - self.ssim_mod(pred_recon, target, data_range=1.0)
        loss_mse = self.mse(pred_recon, target)
        loss_main = self.omega * loss_ssim + (1 - self.omega) * loss_mse

        # 2. 亮度匹配：重建图与目标图均值相等
        brightness_loss = torch.abs(pred_recon.mean() - target.mean())

        return loss_main + self.bw * brightness_loss
