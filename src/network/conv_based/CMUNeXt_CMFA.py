import torch
import torch.nn as nn


# ---------------- 原有基础模块保持不变 ----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
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


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


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
        x = self.conv(x)
        return x


# ---------------- 创新模块：卷积多焦点注意力 (CMFA) ----------------

class CMFA(nn.Module):
    """
    根据文档 5.3 节重构的卷积多焦点注意力层 (Convolutional Multi-Focal Attention)
    用于在解码器融合后提取全局平均池化(平滑特征)与全局最大池化(尖锐特征)，抑制伪影。
    """

    def __init__(self, channels, reduction=16):
        super(CMFA, self).__init__()
        # 确保中间降维通道数至少为 1
        mid_channels = max(1, channels // reduction)

        # 多焦点池化操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的权重多层感知机 (使用 1x1 卷积实现)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )

        # Sigmoid 激活生成联合权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 提取平均响应 (平缓区域结构一致性) 和 最大响应 (恶性毛刺尖端锚点)
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # 融合多焦点权重并激活
        weight = self.sigmoid(avg_out + max_out)

        # 重新调制输出的通道响应分布
        return x * weight


# ---------------- 主网络架构：CMUNeXt_CMFA ----------------

class CMUNeXt_CMFA(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_CMFA, self).__init__()
        # Encoder (保持原版设计)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder 基础融合模块
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])

        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])

        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])

        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        # ---------------- 新增：为每个解码输出阶段实例化 CMFA 模块 ----------------
        self.cmfa5 = CMFA(channels=dims[3])
        self.cmfa4 = CMFA(channels=dims[2])
        self.cmfa3 = CMFA(channels=dims[1])
        self.cmfa2 = CMFA(channels=dims[0])

    def forward(self, x):
        # 编码器传播
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

        # ---------------- 解码器与 CMFA 精修重塑 ----------------

        # Stage 5
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.cmfa5(d5)  # 附加 CMFA 抑制伪影

        # Stage 4
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.cmfa4(d4)  # 附加 CMFA 抑制伪影

        # Stage 3
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.cmfa3(d3)  # 附加 CMFA 抑制伪影

        # Stage 2
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.cmfa2(d2)  # 附加 CMFA 抑制伪影

        # 分类输出头
        d1 = self.Conv_1x1(d2)

        return d1


# ---------------- 模型构建接口 ----------------

def cmunext_cmfa(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_CMFA(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


def cmunext_cmfa_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1],
                   kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt_CMFA(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


def cmunext_cmfa_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3],
                   kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_CMFA(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


if __name__ == '__main__':
    # 张量连通性测试
    model = cmunext_cmfa(input_channel=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("CMUNeXt-CMFA Output shape:", output.shape)  # 预期输出: [2, 1, 256, 256]