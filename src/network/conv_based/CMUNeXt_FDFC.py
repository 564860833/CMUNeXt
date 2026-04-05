import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 基础模块（与原始 CMUNeXt 一致）
# ======================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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
            nn.Upsample(scale_factor=2, mode='bilinear'),
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
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


# ======================================================================
# 创新模块：FDFC（频域特征校准）—— 修正版
# ======================================================================

class FrequencyDomainFeatureCalibration(nn.Module):
    """
    频域特征校准模块 (FDFC) —— 修正版

    修正内容：
    [Fix-1] enhance_factor 维度对齐：
            门控输出统一为 (B, C)，用 .view(B, C, 1, 1) 广播到频谱 (B, C, H', W')，
            不再出现 5 维张量。

    [Fix-2] 频率掩码坐标系：
            rfft2 不做 fftshift，DC 分量在 [0, 0] 位置。
            使用 torch.fft.fftfreq（行方向）和 torch.fft.rfftfreq（列方向）
            获取每个频点的真实归一化频率，再计算到原点的距离。

    [Fix-3] 可学习频率阈值的可微性：
            硬比较 `dist < threshold` 不传梯度。
            改为软 sigmoid 掩码：low_mask = σ((threshold - dist_norm) / τ)，
            τ 为温度参数（默认 0.05），全程可微，梯度平滑。

    [Fix-4] 门控从频域统计计算：
            高频门控由高频幅度的逐通道均方能量驱动：
              high_energy = mean(|high_mag|²)  →  MLP  →  sigmoid  →  门控
            低频门控同理。
            不再从空间域 x 直接计算，保证"频域统计决定频域校准"的一致性。
    """

    def __init__(self, channels, reduction=4, temperature=0.05):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.temperature = temperature

        # 可学习频率分割阈值（通过 Fix-3 的软掩码保证可微）
        self.freq_threshold = nn.Parameter(torch.tensor(0.5))

        # [Fix-4] 高频校准：从高频幅度能量统计计算通道级门控
        self.high_freq_gate = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

        # [Fix-4] 低频增强：从低频幅度能量统计计算通道级门控
        self.low_freq_gate = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

        # 残差缩放因子
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _build_soft_freq_mask(self, spatial_h, spatial_w, device):
        """
        [Fix-2] 构建正确坐标系下的软频率掩码

        rfft2 输出布局：
          - 行方向 (dim=-2)：频率为 fftfreq(spatial_h) = [0, 1/H, ..., (H/2-1)/H, -H/2/H, ..., -1/H]
          - 列方向 (dim=-1)：频率为 rfftfreq(spatial_w) = [0, 1/W, ..., (W//2)/W]
          DC 在 [0, 0]，不在中心。

        [Fix-3] 软掩码：
          low_mask = σ((threshold - dist_norm) / τ)
          全程可微，freq_threshold 可接收梯度。
        """
        # 获取正确的频率坐标
        fy = torch.fft.fftfreq(spatial_h, device=device)   # (freq_h,) 范围 ≈ [-0.5, 0.5)
        fx = torch.fft.rfftfreq(spatial_w, device=device)   # (freq_w,) 范围 [0, 0.5]

        # 构建 2D 频率幅值网格
        yy, xx = torch.meshgrid(fy, fx, indexing='ij')
        dist = torch.sqrt(yy ** 2 + xx ** 2)                # (freq_h, freq_w)

        # 归一化到 [0, 1]
        max_dist = dist.max() + 1e-6
        dist_norm = dist / max_dist

        # 软 sigmoid 分割（可微）
        threshold = torch.sigmoid(self.freq_threshold)
        low_mask = torch.sigmoid((threshold - dist_norm) / self.temperature)
        high_mask = 1.0 - low_mask

        # 扩展为 (1, 1, freq_h, freq_w)，广播到 (B, C, freq_h, freq_w)
        return low_mask.unsqueeze(0).unsqueeze(0), high_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # ---- 1. FFT 变换到频域 ----
        x_freq = torch.fft.rfft2(x, norm='ortho')           # (B, C, H, W//2+1) complex
        magnitude = torch.abs(x_freq)                        # (B, C, H, W//2+1)
        phase = torch.angle(x_freq)                          # (B, C, H, W//2+1)

        # ---- 2. [Fix-2][Fix-3] 软频率分割 ----
        low_mask, high_mask = self._build_soft_freq_mask(H, W, x.device)

        high_mag = magnitude * high_mask                     # (B, C, freq_h, freq_w)
        low_mag = magnitude * low_mask                       # (B, C, freq_h, freq_w)

        # ---- 3. [Fix-4] 高频校准：从高频能量统计计算门控 ----
        high_energy = (high_mag ** 2).mean(dim=(-2, -1))     # (B, C)
        high_gate = self.high_freq_gate(high_energy)          # (B, C)
        # [Fix-1] view 为 (B, C, 1, 1)，正确广播到 4D 频谱
        high_mag_calibrated = high_mag * high_gate.view(B, C, 1, 1)

        # ---- 4. [Fix-4] 低频增强：从低频能量统计计算门控 ----
        low_energy = (low_mag ** 2).mean(dim=(-2, -1))       # (B, C)
        low_gate = self.low_freq_gate(low_energy)             # (B, C)
        # [Fix-1] view 为 (B, C, 1, 1)
        low_mag_enhanced = low_mag * (1.0 + 0.1 * low_gate.view(B, C, 1, 1))

        # ---- 5. 合并 + IFFT ----
        calibrated_mag = high_mag_calibrated + low_mag_enhanced
        x_freq_calibrated = calibrated_mag * torch.exp(1j * phase)
        x_calibrated = torch.fft.irfft2(x_freq_calibrated, s=(H, W), norm='ortho')

        # ---- 6. 残差连接 ----
        return identity + torch.tanh(self.scale) * x_calibrated


# ======================================================================
# CMUNeXt + FDFC 主网络
# ======================================================================

class CMUNeXt_FDFC(nn.Module):
    """
    CMUNeXt + 频域特征校准 (FDFC)

    在深层编码器输出 (x3, x4, x5) 后各添加一个 FDFC 模块。
    浅层 (x1, x2) 分辨率高，FFT 计算开销大且浅层散斑问题不严重，故不添加。
    """

    def __init__(self, input_channel=3, num_classes=1,
                 dims=[16, 32, 128, 160, 256],
                 depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_FDFC, self).__init__()

        # ---- 编码器（与原始 CMUNeXt 一致）----
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # ---- FDFC：深层编码器特征校准 ----
        self.fdfc3 = FrequencyDomainFeatureCalibration(dims[2])
        self.fdfc4 = FrequencyDomainFeatureCalibration(dims[3])
        self.fdfc5 = FrequencyDomainFeatureCalibration(dims[4])

        # ---- 解码器（与原始 CMUNeXt 一致）----
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
        # ---- 编码器 ----
        x1 = self.stem(x)
        x1 = self.encoder1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3 = self.fdfc3(x3)          # ← FDFC 频域校准

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4 = self.fdfc4(x4)          # ← FDFC 频域校准

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)
        x5 = self.fdfc5(x5)          # ← FDFC 频域校准

        # ---- 解码器（与原始 CMUNeXt 一致）----
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

        d1 = self.Conv_1x1(d2)
        return d1


# ======================================================================
# [Fix-5] FDFC 配套损失函数 —— 支持日志字典
# ======================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FDFCLoss(nn.Module):
    """
    CMUNeXt_FDFC 配套损失函数

    = Dice Loss + BCE Loss + λ_freq × 频域一致性正则

    [Fix-5] 支持 return_dict 模式，返回各子损失用于日志记录，
            与项目中 BoundaryDeepSupervisionLoss 等风格一致。

    使用：
        criterion = FDFCLoss(freq_reg_weight=0.1)

        # 模式1：只返回总损失（默认）
        loss = criterion(pred, gt_mask)

        # 模式2：返回总损失 + 子损失字典
        loss, log_dict = criterion(pred, gt_mask, return_dict=True)
        # log_dict = {'dice_loss': ..., 'bce_loss': ..., 'freq_loss': ..., 'total_loss': ...}
    """

    def __init__(self, freq_reg_weight=0.1, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.freq_reg_weight = freq_reg_weight

    def freq_consistency_loss(self, pred, target):
        """频域一致性：预测和GT的频谱幅度应接近"""
        pred_sig = torch.sigmoid(pred)
        pred_freq = torch.fft.rfft2(pred_sig, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_freq), torch.abs(target_freq))

    def forward(self, pred, gt_mask, return_dict=False):
        dice_l = self.dice(pred, gt_mask)
        bce_l = self.bce(pred, gt_mask)
        seg_loss = dice_l + bce_l

        if self.freq_reg_weight > 0:
            freq_l = self.freq_consistency_loss(pred, gt_mask)
            total_loss = seg_loss + self.freq_reg_weight * freq_l
        else:
            freq_l = torch.tensor(0.0, device=pred.device)
            total_loss = seg_loss

        if return_dict:
            log_dict = {
                'dice_loss': dice_l.item(),
                'bce_loss': bce_l.item(),
                'freq_loss': freq_l.item(),
                'total_loss': total_loss.item(),
            }
            return total_loss, log_dict

        return total_loss


# ======================================================================
# 工厂函数
# ======================================================================

def cmunext_fdfc(input_channel=3, num_classes=1):
    return CMUNeXt_FDFC(
        input_channel=input_channel, num_classes=num_classes,
        dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],
    )

def cmunext_fdfc_s(input_channel=3, num_classes=1):
    return CMUNeXt_FDFC(
        input_channel=input_channel, num_classes=num_classes,
        dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9],
    )

def cmunext_fdfc_l(input_channel=3, num_classes=1):
    return CMUNeXt_FDFC(
        input_channel=input_channel, num_classes=num_classes,
        dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7],
    )
