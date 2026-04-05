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
#  机制: 噪声预测 + 加法减除
# ═══════════════════════════════════════════════

class DDSR(nn.Module):
    def __init__(self, channels, smooth_k=7):
        super().__init__()

        self.smooth = nn.AvgPool2d(
            kernel_size=smooth_k, stride=1, padding=smooth_k // 2
        )

        self.lin_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.log_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.noise_pred = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

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
#  DualGAG — Dual Gated Attention Gate
#  机制: 解码器语义引导的乘法门控
# ═══════════════════════════════════════════════

class DualGatedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, groups=4, reduction=8):
        super().__init__()
        actual_groups = groups if (F_int % groups == 0 and F_l % groups == 0) else 1
        self.groups = actual_groups

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, stride=1, padding=1,
                      groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
            nn.GELU(),
        )
        self.spatial_gate = nn.Conv2d(F_int, actual_groups, kernel_size=1,
                                      stride=1, padding=0, bias=True)
        self.spatial_scale = nn.Parameter(torch.full((actual_groups,), 0.1))

        channel_mid = max(8, F_int // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(F_int, channel_mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_mid, F_l, kernel_size=1, bias=False),
        )
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        fused = self.relu(self.W_g(g) + self.W_x(x))
        fused = self.refine(fused)

        spatial_gate = torch.sigmoid(self.spatial_gate(fused))
        spatial_gate = 2.0 * spatial_gate - 1.0
        spatial_gate = 1.0 + torch.tanh(self.spatial_scale).view(
            1, self.groups, 1, 1
        ) * spatial_gate

        if self.groups == 1:
            spatial_mod = spatial_gate
        else:
            spatial_mod = []
            for gi, chunk in enumerate(torch.chunk(x, self.groups, dim=1)):
                spatial_mod.append(
                    spatial_gate[:, gi:gi + 1].expand(-1, chunk.size(1), -1, -1)
                )
            spatial_mod = torch.cat(spatial_mod, dim=1)

        channel_gate = (
            self.channel_mlp(self.avg_pool(fused))
            + self.channel_mlp(self.max_pool(fused))
        )
        channel_gate = torch.sigmoid(channel_gate)
        channel_gate = 2.0 * channel_gate - 1.0
        channel_gate = 1.0 + torch.tanh(self.channel_scale) * channel_gate

        return x * spatial_mod * channel_gate


# ═══════════════════════════════════════════════
#  CMUNeXt_SpeckleEnhance_DualGAG
#
#  数据流:
#
#  Encoder → DDSR (x − α·noise) → skip → DualGAG (x × gate) → Decoder
#            ^^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^
#            加法纠正 (编码端)              乘法选择 (解码端)
#            "先净化"                      "再筛选"
# ═══════════════════════════════════════════════

class CMUNeXt_SpeckleEnhance_DualGAG(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
        ddsr_smooth_k=7,
    ):
        super().__init__()
        self.ddsr_stages = set(ddsr_stages)
        self.gag_stages = set(gag_stages)

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

        # ── DDSR (编码主路径, 加法纠正) ──
        skip_dims = [dims[0], dims[1], dims[2], dims[3]]
        self.ddsr_modules = nn.ModuleDict()
        for s in ddsr_stages:
            self.ddsr_modules[str(s)] = DDSR(
                channels=skip_dims[s], smooth_k=ddsr_smooth_k
            )

        # ── DualGAG (skip-decoder 交汇, 乘法选择) ──
        self.gag_modules = nn.ModuleDict()
        if 3 in self.gag_stages:
            self.gag_modules["3"] = DualGatedAttentionGate(
                F_g=dims[3], F_l=dims[3],
                F_int=max(8, dims[3] // 2), groups=4,
            )
        if 2 in self.gag_stages:
            self.gag_modules["2"] = DualGatedAttentionGate(
                F_g=dims[2], F_l=dims[2],
                F_int=max(8, dims[2] // 2), groups=4,
            )
        if 1 in self.gag_stages:
            self.gag_modules["1"] = DualGatedAttentionGate(
                F_g=dims[1], F_l=dims[1],
                F_int=max(8, dims[1] // 2), groups=4,
            )
        if 0 in self.gag_stages:
            self.gag_modules["0"] = DualGatedAttentionGate(
                F_g=dims[0], F_l=dims[0],
                F_int=max(8, dims[0] // 2), groups=2,
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

    def _apply_gag(self, g, x, stage):
        if stage in self.gag_stages:
            return self.gag_modules[str(stage)](g=g, x=x)
        return x

    def forward(self, x):
        # ════ Encode + DDSR 净化 ════
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

        # ════ Decode + DualGAG 筛选 ════
        d5 = self.Up5(x5)
        x4_p = self._apply_gag(d5, x4, 3)
        d5 = self.Up_conv5(torch.cat((x4_p, d5), dim=1))

        d4 = self.Up4(d5)
        x3_p = self._apply_gag(d4, x3, 2)
        d4 = self.Up_conv4(torch.cat((x3_p, d4), dim=1))

        d3 = self.Up3(d4)
        x2_p = self._apply_gag(d3, x2, 1)
        d3 = self.Up_conv3(torch.cat((x2_p, d3), dim=1))

        d2 = self.Up2(d3)
        x1_p = self._apply_gag(d2, x1, 0)
        d2 = self.Up_conv2(torch.cat((x1_p, d2), dim=1))

        return self.Conv_1x1(d2)


# ═══════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════

def cmunext_speckle_dualgag(input_channel=3, num_classes=1):
    """推荐: DDSR 浅层 + DualGAG 深层"""
    return CMUNeXt_SpeckleEnhance_DualGAG(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
    )

def cmunext_speckle_dualgag_full(input_channel=3, num_classes=1):
    """全配置: 两者均在所有阶段"""
    return CMUNeXt_SpeckleEnhance_DualGAG(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1, 2, 3),
        gag_stages=(0, 1, 2, 3),
    )

def cmunext_speckle_dualgag_light(input_channel=3, num_classes=1):
    """轻量: DDSR 浅层 + DualGAG 深层"""
    return CMUNeXt_SpeckleEnhance_DualGAG(
        input_channel=input_channel, num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
    )

def cmunext_speckle_dualgag_s(input_channel=3, num_classes=1):
    return CMUNeXt_SpeckleEnhance_DualGAG(
        input_channel=input_channel, num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
    )

def cmunext_speckle_dualgag_l(input_channel=3, num_classes=1):
    return CMUNeXt_SpeckleEnhance_DualGAG(
        input_channel=input_channel, num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
    )
