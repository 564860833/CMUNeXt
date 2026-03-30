import torch
import torch.nn as nn


# ---------------- 原有基础模块保持不变 ----------------

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
            nn.Upsample(scale_factor=2, mode='bilinear'),
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


# ---------------- 创新模块：多核深度卷积 (MKDC) ----------------

class MKDCBlock(nn.Module):
    """
    根据文档 5.1 节重构的多核倒残差模块
    """

    def __init__(self, ch_in, expansion=4):
        super(MKDCBlock, self).__init__()
        hidden_dim = int(ch_in * expansion)

        # 1. 特征预处理与膨胀
        self.expand_conv = nn.Sequential(
            nn.Conv2d(ch_in, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # 2. 多核深度卷积阵列并联 (MKDC Integration)
        # 分支1: 3x3
        self.dw_conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        # 分支2: 5x5
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        # 分支3: 7x7
        self.dw_conv7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # 3. 特征投影与残差闭环
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, ch_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_in)
        )

    def forward(self, x):
        identity = x

        # 膨胀映射至高维空间
        hidden = self.expand_conv(x)

        # 多尺度特征并行提取与聚合 (逐元素相加)
        out3 = self.dw_conv3(hidden)
        out5 = self.dw_conv5(hidden)
        out7 = self.dw_conv7(hidden)
        out = out3 + out5 + out7

        # 降维投影与残差相加
        out = self.project_conv(out)
        return out + identity


class CMUNeXtBlock_MKDC(nn.Module):
    """
    重构的编码器基础块，使用 MKDC 替代原有的单一大核
    """

    def __init__(self, ch_in, ch_out, depth=1):
        super(CMUNeXtBlock_MKDC, self).__init__()
        self.block = nn.Sequential(
            *[MKDCBlock(ch_in=ch_in, expansion=4) for _ in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


# ---------------- 主网络架构：CMUNeXt_MKDC ----------------

class CMUNeXt_MKDC(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1]):
        """
        注：移除了原版中的 kernels 参数，因为 MKDC 固定内置了 [3, 5, 7] 并行多核阵列。
        """
        super(CMUNeXt_MKDC, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])

        # 使用全新的 MKDC Block 进行编码特征提取
        self.encoder1 = CMUNeXtBlock_MKDC(ch_in=dims[0], ch_out=dims[0], depth=depths[0])
        self.encoder2 = CMUNeXtBlock_MKDC(ch_in=dims[0], ch_out=dims[1], depth=depths[1])
        self.encoder3 = CMUNeXtBlock_MKDC(ch_in=dims[1], ch_out=dims[2], depth=depths[2])
        self.encoder4 = CMUNeXtBlock_MKDC(ch_in=dims[2], ch_out=dims[3], depth=depths[3])
        self.encoder5 = CMUNeXtBlock_MKDC(ch_in=dims[3], ch_out=dims[4], depth=depths[4])

        # Decoder (保持原版设计以兼容，若需进一步增强可加入GAG模块)
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


# ---------------- 模型构建接口 ----------------

def cmunext_mkdc(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1]):
    return CMUNeXt_MKDC(dims=dims, depths=depths, input_channel=input_channel, num_classes=num_classes)


def cmunext_mkdc_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1]):
    return CMUNeXt_MKDC(dims=dims, depths=depths, input_channel=input_channel, num_classes=num_classes)


def cmunext_mkdc_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3]):
    return CMUNeXt_MKDC(dims=dims, depths=depths, input_channel=input_channel, num_classes=num_classes)


if __name__ == '__main__':
    # 测试模型流转与张量维度
    model = cmunext_mkdc(input_channel=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 预期: [2, 1, 256, 256]
