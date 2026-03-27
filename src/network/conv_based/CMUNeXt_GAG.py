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


# ---------------- 创新模块：分组注意力门 (GAG) ----------------

class GroupedAttentionGate(nn.Module):
    """
    根据文档 5.2 节重构的分组注意力门模块 (GAG)
    """

    def __init__(self, F_g, F_l, F_int, groups=4):
        super(GroupedAttentionGate, self).__init__()

        # 确保 groups 能够整除中间通道数 F_int，否则默认回退为普通卷积 (groups=1)
        actual_groups = groups if F_int % groups == 0 else 1

        # 门控信号处理 (来自深层解码器): 3x3 分组卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 浅层特征处理 (来自跳跃连接): 3x3 分组卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 注意力系数投影: ReLU 激活后，通过 1x1 卷积压缩通道，再使用 Sigmoid 激活
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 提取空间结构特征
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 逐元素相加融合
        psi = self.relu(g1 + x1)

        # 映射为 0~1 的空间-通道注意力系数矩阵 (Alpha)
        psi = self.psi(psi)

        # 软过滤：将注意力权重作用于原始浅层特征
        x_purified = x * psi
        return x_purified


# ---------------- 主网络架构：CMUNeXt_GAG ----------------

class CMUNeXt_GAG(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GAG, self).__init__()
        # Encoder (保持原版大核架构)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder 基础模块
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])

        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])

        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])

        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        # ---------------- 新增：为每个解码阶段实例化 GAG 模块 ----------------
        # 中间投影通道数 F_int 设为输入通道数的一半，以降低参数量
        self.gag5 = GroupedAttentionGate(F_g=dims[3], F_l=dims[3], F_int=dims[3] // 2, groups=4)
        self.gag4 = GroupedAttentionGate(F_g=dims[2], F_l=dims[2], F_int=dims[2] // 2, groups=4)
        self.gag3 = GroupedAttentionGate(F_g=dims[1], F_l=dims[1], F_int=dims[1] // 2, groups=4)
        self.gag2 = GroupedAttentionGate(F_g=dims[0], F_l=dims[0], F_int=dims[0] // 2, groups=2)  # 浅层通道少，降低groups数

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

        # ---------------- 解码器与 GAG 软过滤融合 ----------------

        # Stage 5
        d5 = self.Up5(x5)  # d5 作为门控信号 (g)
        x4_purified = self.gag5(g=d5, x=x4)  # 使用 GAG 净化浅层特征 x4
        d5 = torch.cat((x4_purified, d5), dim=1)  # 纯净特征与上采样特征拼接
        d5 = self.Up_conv5(d5)  # 送入 fusion_conv 平滑融合

        # Stage 4
        d4 = self.Up4(d5)
        x3_purified = self.gag4(g=d4, x=x3)
        d4 = torch.cat((x3_purified, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # Stage 3
        d3 = self.Up3(d4)
        x2_purified = self.gag3(g=d3, x=x2)
        d3 = torch.cat((x2_purified, d3), dim=1)
        d3 = self.Up_conv3(d3)

        # Stage 2
        d2 = self.Up2(d3)
        x1_purified = self.gag2(g=d2, x=x1)
        d2 = torch.cat((x1_purified, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 分类输出头
        d1 = self.Conv_1x1(d2)

        return d1


# ---------------- 模型构建接口 ----------------

def cmunext_gag(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GAG(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


def cmunext_gag_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1],
                  kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt_GAG(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


def cmunext_gag_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3],
                  kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GAG(dims=dims, depths=depths, kernels=kernels, input_channel=input_channel, num_classes=num_classes)


if __name__ == '__main__':
    # 张量测试
    model = cmunext_gag(input_channel=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("CMUNeXt-GAG Output shape:", output.shape)  # 预期输出: [2, 1, 256, 256]