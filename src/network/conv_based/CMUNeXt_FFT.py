import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件 (复用自 CMUNeXt 原始设计)
# ==========================================
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 2. 核心创新模块: 自适应高斯频域滤波器
# ==========================================
class LearnableGaussianFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 💡 核心优化 1: 每个通道只学习一个参数(截止频率 sigma)，彻底杜绝小数据集过拟合
        # 修正初始化：计算 softplus 的逆函数，确保初始时的实际截止频率精确为 0.2
        # inv_softplus(0.2) = log(exp(0.2) - 1) ≈ -1.5076
        target_sigma = 0.2
        init_val = math.log(math.exp(target_sigma) - 1.0)

        self.sigma = nn.Parameter(torch.ones(1, dim, 1, 1) * init_val)

        # 💡 核心优化 2: 残差输出零初始化，确保 Epoch 0 时的平稳启动 (Identity Mapping)
        self.out_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 傅里叶正变换 (转入频域)
        fft_x = torch.fft.rfft2(x, norm='ortho')

        # 2. 动态生成精确的物理频率网格 (无视输入尺寸，彻底解决插值导致的高低频错乱)
        device = x.device
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)

        # 生成 2D 频率坐标网格
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')

        # 计算每个频点到中心的距离的平方： D^2(u, v) = u^2 + v^2
        D2 = (FX ** 2 + FY ** 2).unsqueeze(0).unsqueeze(0)

        # 3. 构造自适应高斯低通滤波器: H(u,v) = exp(- D^2 / (2 * sigma^2))
        # 使用 softplus 确保 sigma 永远为正数，加上 epsilon 防止除零
        # 此时 Epoch 0 的 sigma_safe 将精确等于 0.2 + 1e-5
        sigma_safe = F.softplus(self.sigma) + 1e-5
        gaussian_filter = torch.exp(-D2 / (2 * sigma_safe ** 2))

        # 4. 频域滤波：复数频谱乘上实数衰减权重
        filtered_fft_x = fft_x * gaussian_filter

        # 5. 傅里叶逆变换 (转回空域)
        out = torch.fft.irfft2(filtered_fft_x, s=(H, W), norm='ortho')

        return out * self.out_scale


# ==========================================
# 3. 架构组件: 严格遵循现代标准的 FFT Block
# ==========================================
class OptimizedFFTBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        # 💡 核心优化 3: 严格的 Pre-Norm 机制
        self.norm1 = nn.BatchNorm2d(ch_in)

        # 空间混合：自适应高斯频域滤波
        self.fft = LearnableGaussianFilter(ch_in)

        # 💡 核心优化 4: 空间与通道彻底解耦，独立的 FFN 与残差
        self.norm2 = nn.BatchNorm2d(ch_in)
        self.ffn = nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ch_in * 4, ch_in, kernel_size=1)
        )

        # 对 FFN 的最后一层也做零初始化，确保深层网络的梯度完美穿透
        nn.init.constant_(self.ffn[-1].weight, 0)
        nn.init.constant_(self.ffn[-1].bias, 0)

    def forward(self, x):
        # 现代网络标准残差公式: x = x + Mixer(Norm(x))
        x = x + self.fft(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FFTSpatialStage(nn.Module):
    """管理每个 Stage 中多个 FFT Block 以及最后的通道调整"""

    def __init__(self, ch_in, ch_out, depth):
        super().__init__()
        # 构建深度堆叠的 FFT Block
        self.blocks = nn.ModuleList([
            OptimizedFFTBlock(ch_in) for _ in range(depth)
        ])
        # 保持 CMUNeXt 的通道扩张/压缩逻辑
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.up(x)
        return x


# ==========================================
# 4. 主网络: CMUNeXt_FFT_Optimized
# ==========================================
class CMUNeXt_FFT(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1]):
        super(CMUNeXt_FFT, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])

        # 替换为我们极度简洁、无视尺寸、免疫过拟合的 FFTSpatialStage
        self.encoder1 = FFTSpatialStage(ch_in=dims[0], ch_out=dims[0], depth=depths[0])
        self.encoder2 = FFTSpatialStage(ch_in=dims[0], ch_out=dims[1], depth=depths[1])
        self.encoder3 = FFTSpatialStage(ch_in=dims[1], ch_out=dims[2], depth=depths[2])
        self.encoder4 = FFTSpatialStage(ch_in=dims[2], ch_out=dims[3], depth=depths[3])
        self.encoder5 = FFTSpatialStage(ch_in=dims[3], ch_out=dims[4], depth=depths[4])

        # Decoder (保持原始高效的上采样设计)
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])

        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])

        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])

        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder 提取特征 (融入物理频域去噪)
        x1 = self.stem(x)
        x1 = self.encoder1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # Decoder 与跳跃连接 (Skip Connections)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        return out

