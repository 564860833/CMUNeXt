import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 基础组件 (从 CMUNeXt 复用)
# --------------------------

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
            nn.ReLU(inplace=True) # 这里保留 ReLU 不影响整体，也可全换 GELU
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
        layers = []
        for _ in range(depth):
            layers.append(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                ))
            )
            layers.append(nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm2d(ch_in * 4))
            layers.append(nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm2d(ch_in))

        self.block = nn.Sequential(*layers)
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

# --------------------------
# 核心创新模块: GMS_ASPP (门控多尺度抗噪融合)
# --------------------------

class GMS_ASPP(nn.Module):
    """
    针对乳腺超声优化的 Gated Multi-scale Noise-Suppression ASPP
    全部统一使用 GELU，且引入 3x3 门控卷积增强空间感知。
    """
    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 2, 3, 4), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()

        # Branch 1: 1x1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ))

        # Branches 2-5: atrous conv
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))

        # Branch 6: image pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

        self.n_branches = len(dilations) + 2

        # 核心：上下文感知的门控生成器 (Context-Aware Gate Generation)
        self.gate_generator = nn.Sequential(
            # 使用 3x3 卷积，赋予空间感知能力，更好识别超声中的局部散斑噪声
            nn.Conv2d(out_ch, out_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, self.n_branches, kernel_size=1, bias=False),
            nn.Sigmoid() # 将掩码权重约束在 0~1 之间
        )

        # 最终特征对齐输出
        self.project = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]

        # 1. 提取各尺度特征
        feats = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)

        # 2. 上下文感知：将特征进行融合预判 (F_sum)
        f_sum = sum(feats)

        # 3. 生成空间门控掩码, shape: [B, n_branches, H, W]
        gates = self.gate_generator(f_sum)

        # 4. 门控调制与抗噪融合 (替代 cat)
        out_fused = torch.zeros_like(f_sum)
        for i in range(self.n_branches):
            # 提取第 i 个分支的专属门控权重 [B, 1, H, W]
            gate_i = gates[:, i:i+1, :, :]
            # 过滤噪声后相加
            out_fused = out_fused + (feats[i] * gate_i)

        # 5. 最终特征对齐
        out = self.project(out_fused)
        return out


# --------------------------
# 主网络: CMUNeXt 搭载 GMS_ASPP
# --------------------------

class CMUNeXt_GMSF_ASPP(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=[16, 32, 128, 160, 256],
        depths=[1, 1, 1, 3, 1],
        kernels=[3, 3, 7, 7, 7],
        aspp_dilations=(1, 2, 3, 4),
        aspp_dropout=0.1,
    ):
        super(CMUNeXt_GMSF_ASPP, self).__init__()

        # Encoder 初始化
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # 在瓶颈层插入门控抗噪的 ASPP
        self.aspp = GMS_ASPP(in_ch=dims[4], out_ch=dims[4], dilations=aspp_dilations, dropout=aspp_dropout)

        # Decoder 初始化
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
        # Stage 1
        x1 = self.stem(x)
        x1 = self.encoder1(x1)

        # Stage 2
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)

        # Stage 3
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)

        # Stage 4
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)

        # Stage 5 (bottleneck)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # ASPP (GMSF) 门控抗噪特征增强
        x5 = self.aspp(x5)

        # Decoder
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