import torch
import torch.nn as nn


# ═══════════════════════════════════════════════
#  基础模块 (与 CMUNeXt 一致)
# ═══════════════════════════════════════════════

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.up(x)


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )
    def forward(self, x):
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=k, groups=ch_in, padding=k // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            ) for _ in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)
    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


# ═══════════════════════════════════════════════
#  DDSR — Dual-Domain Speckle Refinement
# ═══════════════════════════════════════════════

class DDSR(nn.Module):
    """
    Dual-Domain Speckle Refinement

    超声 speckle 服从乘性噪声模型 I = S × N。
    线性域残差保留边界的绝对强度变化；
    对数域残差将乘性噪声化为加性，便于线性分离。
    融合双域特征后直接预测噪声分量，做加法减除。

    与门控式注意力的本质区别:
      门控: output = x × gate          乘法选择, 只能衰减
      减除: output = x − α · noise     加法纠正, 可恢复被偏移的真值
    """

    def __init__(self, channels, smooth_k=7):
        super().__init__()

        # ── 结构估计 ──
        self.smooth = nn.AvgPool2d(
            kernel_size=smooth_k, stride=1, padding=smooth_k // 2
        )

        # ── 线性域残差分支 ──
        self.lin_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # ── 对数域残差分支 ──
        self.log_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # ── 噪声预测器 ──
        self.noise_pred = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # ── 零初始化残差系数 ──
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def signed_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, x):
        struct = self.smooth(x)
        detail = x - struct

        lin_feat = self.lin_branch(detail)

        x_log = self.signed_log(x)
        struct_log = self.smooth(x_log)
        log_detail = x_log - struct_log
        log_feat = self.log_branch(log_detail)

        noise = self.noise_pred(torch.cat([lin_feat, log_feat], dim=1))

        return x - self.alpha * noise


# ═══════════════════════════════════════════════
#  CMUNeXt_SpeckleEnhance
#
#  相对 CMUNeXt 的唯一改动:
#    编码器每个 stage 输出后接 DDSR，净化 skip 特征
#    同时净化流向下游编码器的输入（级联效应）
#    编码器、解码器结构完全不变
# ═══════════════════════════════════════════════

class CMUNeXt_SpeckleEnhance(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1, 2, 3),
        ddsr_smooth_k=7,
    ):
        super().__init__()
        self.ddsr_stages = set(ddsr_stages)

        # ── Encoder (与 CMUNeXt 完全一致) ──
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0],
                                     depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1],
                                     depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2],
                                     depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3],
                                     depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4],
                                     depth=depths[4], k=kernels[4])

        # ── DDSR 模块 ──
        skip_dims = [dims[0], dims[1], dims[2], dims[3]]
        self.ddsr_modules = nn.ModuleDict()
        for s in ddsr_stages:
            self.ddsr_modules[str(s)] = DDSR(
                channels=skip_dims[s], smooth_k=ddsr_smooth_k
            )

        # ── Decoder (与 CMUNeXt 完全一致) ──
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def _apply_ddsr(self, x, stage):
        if stage in self.ddsr_stages:
            return self.ddsr_modules[str(stage)](x)
        return x

    def forward(self, x):
        # ════ Encode + DDSR ════
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x1 = self._apply_ddsr(x1, 0)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x2 = self._apply_ddsr(x2, 1)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3 = self._apply_ddsr(x3, 2)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4 = self._apply_ddsr(x4, 3)

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # ════ Decode (与 CMUNeXt 完全一致) ════
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        return self.Conv_1x1(d2)


# ═══════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════

def cmunext_speckle(input_channel=3, num_classes=1):
    """默认: 全 4 阶段 DDSR"""
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1, 2, 3),
    )

def cmunext_speckle_shallow(input_channel=3, num_classes=1):
    """仅浅层 DDSR (stage 0, 1)"""
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
    )

def cmunext_speckle_deep(input_channel=3, num_classes=1):
    """仅深层 DDSR (stage 2, 3)"""
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(2, 3),
    )

def cmunext_speckle_s(input_channel=3, num_classes=1):
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel, num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        ddsr_stages=(0, 1, 2, 3),
    )

def cmunext_speckle_l(input_channel=3, num_classes=1):
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel, num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1, 2, 3),
    )