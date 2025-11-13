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
# class AberrationCNN(nn.Module):
#     def __init__(self,in_ch=3,feat_ch=128):
#         super(AberrationCNN, self).__init__()
#         #卷积层1：3×3卷积核，保持尺寸
#         self.conv1=nn.Conv2d(in_channels=in_ch,out_channels=feat_ch,kernel_size=3,stride=1,padding=1)
#         self.bn1=nn.BatchNorm2d(feat_ch)

#         #卷积层2：细化特征，聚焦像差相关模式
#         self.conv2=nn.Conv2d(in_channels=feat_ch,out_channels=feat_ch,kernel_size=3,stride=1,padding=1)
#         self.bn2=nn.BatchNorm2d(feat_ch)

#         #逆卷积层：复原到原图通道
#         self.deconv=nn.ConvTranspose2d(in_channels=feat_ch,out_channels=in_ch,kernel_size=3,stride=1,padding=1)
#         #激活
#         self.act=nn.ReLU(inplace=True)

#     def forward(self,x):
#         x=self.act(self.bn1(self.conv1(x)))
#         x=self.act(self.bn2(self.conv2(x)))
#         x=self.deconv(x)
#         #输出为预校正图像
#         out=torch.clamp(x,0.0,1)
#         return out

#     def get_features(self, x):
#         """提取conv2后的特征图作为结构特征（用于损失计算）"""
#         x = self.act(self.bn1(self.conv1(x)))
#         x = self.act(self.bn2(self.conv2(x)))
#         return x  # 返回特征图（非输出图像）


class ChannelAttention(nn.Module):
    '''
    来源：SE-Net（Squeeze-and-Excitation Network, 2017）
    作用：让模型 动态决定每个通道（channel）的重要性
    在 AberrationCNN_v3 中的用途：R、G、B 三个通道的像差补偿强度不同 → 让网络自动学习哪个通道需要更强的“预失真”补偿

    核心逻辑：
    1. Squeeze（压缩）：全局平均池化 → 提取每个通道的全局空间信息
    2. Excitation（激励）：全连接层+激活 → 学习通道间依赖关系，输出注意力权重
    3. Fusion（融合）：权重与原特征逐通道相乘 → 强化重要通道
    '''
    # reduction=8，为压缩比：中间全连接层通道数=channels//8
    def __init__(self, channels, reduction=8):
        '''
        输出为加权后的特征图
        '''
        super().__init__()
        # Squeeze操作，进行全局平均池化，将每个通道的[H,W]空间信息压缩为一个标量
        # 输出尺寸为：[B,C,1,1]，从空间维度上提取全局上下信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation操作：两层全连接层(FC)
        # FC1+ReLU，降维压缩、减少参数，提取非线性关系
        # FC2+Sigmoid，恢复通道数，输出0~1的权重（注意力系数）
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), #降维
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), #升维
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播逻辑
        Args:
            x: 输入特征图，shape = [B, C, H, W]（B=批量大小，C=通道数，H=高度，W=宽度）
        Returns:
            加权后的特征图，shape = [B, C, H, W]（与输入维度一致）
        """
        b, c, _, _ = x.size()
        # y为学习到的通道权重
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # 逐个通道加权

# 多尺度特征提取模块
# 
class MultiScaleBlock(nn.Module):
    '''
    多尺度：该模块是“双感受野+注意力”的特征提取器，使其在网络像差校正时能够“看清近处球差，以及远处的慧差”
    输入：[B,in_ch,H,W] 
    输出：[B,out_ch,H,W]
    空间尺寸完全保持不变，通道数从in_ch扩展为out_ch，实现了多尺度融合+通道注意力加权
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 3×3 和 5×5 的卷积并行，可以用于捕捉局部细节+中等范围的像差（球差、慧差）
        # 切割通道，两个分支各占一半
        self.branch1 = nn.Conv2d(in_ch, out_ch//2, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(in_ch, out_ch//2, kernel_size=5, padding=2)

        # 批归一化加速收敛
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # 通道拼接，融合多尺度的特征
        # 结合ChannelAttention的功能可知，是为了给out_ch个通道加权
        self.ca = ChannelAttention(out_ch)

    def forward(self, x):
        # b1与b2并行处理
        b1 = self.act(self.branch1(x))      # 分支1：3×3 卷积 + ReLU
        b2 = self.act(self.branch2(x))      # 分支2：5×5 卷积 + ReLU
        # 前面将其分为了out_ch切分成了两个通道
        out = torch.cat([b1, b2], dim=1)    # 通道拼接
        out = self.bn(out)                  # 批归一化
        out = self.ca(out)                  # 通道注意力加权
        return out


class AberrationCNN_v3(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        # 1. 初始卷积（提取基础特征）
        # 将RGB输入映射到高维特征空间
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        # 2. 多尺度特征提取（3 层）
        self.ms1 = MultiScaleBlock(base_ch, base_ch) # 64 → 64 局部像差
        self.ms2 = MultiScaleBlock(base_ch, base_ch * 2) # 64 → 128 中层像差
        self.ms3 = MultiScaleBlock(base_ch * 2, base_ch * 2) # 128 → 128 全局模式

        # 3. 上采样（解码器） + 残差融合
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2) # 128 → 64，[B,128,13,13] → [B,64,26,26]
        self.up2 = nn.ConvTranspose2d(base_ch, in_ch, 2, stride=2) # 64 → 3，[B,64,26,26] → [B,3,52,52]

        self.skip_conv=nn.Conv2d(base_ch*2,base_ch,kernel_size=1) # 跳连通道对齐，使用1×1的卷积压缩通道，使跳连可以和up1输出相加

        # 4. 最终输出头
        # 再通过一次卷积来修正像差细节
        self.out_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)

        # 5. 残差连接权重
        self.res_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 保留原始图像信息，用于最终残差连接
        identity = x # [1,3,104,104]

        # 第一阶段：下采样（编码器）--提取多尺度像差特征
        x1 = self.stem(x)           # [B,64,104,104]
        x2 = F.avg_pool2d(x1, 2)    # [B,64,52,52]
        x3 = self.ms1(x2)
        x4 = F.avg_pool2d(x3, 2)    # [B,64,26,26]
        x5 = self.ms2(x4)           # [B,128,26,26] ← 保存
        x6 = F.avg_pool2d(x5, 2)    # [B,128,13,13]
        x7 = self.ms3(x6)           # [B,128,13,13]

        # 上采样 + 跳连
        x = self.up1(x7)                    # [B,64,26,26]
        x5_aligned = self.skip_conv(x5)     # [B,128,26,26] → [B,64,26,26]
        x = x + x5_aligned                  # 通道一致！

        x = self.up2(x)                     # [B,3,52,52]
        x = F.interpolate(x, size=identity.shape[2:], mode='bilinear', align_corners=False)
        x = self.out_conv(x)

        out = torch.clamp(x + self.res_weight * identity, 0.0, 1.0)
        return out
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

# ---------- 1. 损失函数 ----------
class AberrationLossV2(nn.Module):
    def __init__(self, omega=0.7, bright_weight=2.0):
        super().__init__()
        self.omega = omega
        self.bw   = bright_weight
        self.mse  = nn.MSELoss()

    def forward(self, pred_recon, target, pre_corrected=None):
        # 主损失（公式 11）
        loss_ssim = 1 - ssim(pred_recon, target, data_range=1.0)
        loss_mse  = self.mse(pred_recon, target)
        loss_main = self.omega * loss_ssim + (1 - self.omega) * loss_mse

        # 亮度匹配正则（只依赖重建图和目标图）
        loss_bright = torch.abs(pred_recon.mean() - target.mean())

        return loss_main + self.bw * loss_bright
    
class AberrationLossV3(nn.Module):
    """
    pred_recon:  模拟重建图像  [B,3,H,W]   ← 预校正图像 经过 PSF 卷积 后的结果
    target:      原始 clean EI  [B,3,H,W]   ← 理想无像差图像
    pre_corrected: 未使用（保留接口兼容性）
    """
    def __init__(self, 
                 ssim_weight=0.5,      # 降低 SSIM 权重
                 mse_weight=0.3,       # 提升 MSE 权重
                 color_weight=1.0,     # 新增：颜色一致性
                 bright_weight=1.5):   # 亮度匹配
        super().__init__()
        self.sw = ssim_weight
        self.mw = mse_weight
        self.cw = color_weight
        self.bw = bright_weight

    def forward(self, pred_recon, target, pre_corrected=None):
        """
        pred_recon:  模拟重建图像  [B,3,H,W]
        target:      原始 clean EI  [B,3,H,W]
        """
        # 1. SSIM（结构）
        # 关注图像的纹理、边缘、对比度
        loss_ssim = 1 - ssim(pred_recon, target, data_range=1.0)

        # 2. MSE（像素级）
        # 计算像素的平方差
        loss_mse = F.mse_loss(pred_recon, target)

        # 3. 颜色一致性（通道直方图匹配）
        # 颜色一致性损失检查
        # 对 R、G、B 每个通道 分别统计 [0,1] 区间内的直方图（50个bin），归一化为概率分布
        # 用 L1 距离 衡量两个直方图的差异
        # 平均三个通道 → 得到颜色损失
        def channel_hist_loss(x, y):
            loss = 0.0
            for c in range(3):
                hist_x = torch.histc(x[:, c], bins=50, min=0, max=1)
                hist_y = torch.histc(y[:, c], bins=50, min=0, max=1)
                hist_x = hist_x / (hist_x.sum() + 1e-8)
                hist_y = hist_y / (hist_y.sum() + 1e-8)
                loss += F.l1_loss(hist_x, hist_y)
            return loss / 3
        loss_color = channel_hist_loss(pred_recon, target)

        # 4. 亮度匹配
        # 计算整张图的平均像素值
        # 取绝对差值，防止图像整体变量或变暗
        loss_bright = torch.abs(pred_recon.mean() - target.mean())

        # 5. 总损失
        total = (self.sw * loss_ssim +
                 self.mw * loss_mse +
                 self.cw * loss_color +
                 self.bw * loss_bright)

        return total

