import torch
import torch.nn as nn


# --------------------------
# 基础组件 (保持不变)
# --------------------------
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
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


# --------------------------
# 抢救模块 1: 温和版 LKA (Mild-LKA)
# --------------------------
class MildLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 优化点 1: 缩小局部感受野 5x5 -> 3x3
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 优化点 2: 降低空洞率 7x7 (d=3) -> 5x5 (d=2)
        # 有效核大小为 9，padding 为 4。大幅减少深层特征图的 Padding 伪影。
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, groups=dim, dilation=2)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


# --------------------------
# 抢救模块 2: LKASpatialBlock (加入正则化)
# --------------------------
class LKASpatialBlock(nn.Module):
    def __init__(self, dim, k=3, drop_prob=0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=k, groups=dim, padding=k // 2)
        self.lka = MildLKA(dim)

        self.proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.act = nn.GELU()

        # 优化点 3: 引入 Dropout 强行打破过拟合
        self.drop = nn.Dropout2d(drop_prob)

        # 优化点 4: 使用极小值初始化代替绝对的 0 初始化，避免早期训练彻底停滞
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        identity = x
        out = self.norm(x)

        local_feat = self.local_conv(out)
        global_feat = self.lka(out)

        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        fused = self.proj(concat_feat)
        fused = self.act(fused)
        fused = self.drop(fused)  # 应用 Dropout

        return identity + fused


# --------------------------
# 抢救模块 3: LKA-CMU Block (收缩冗余参数)
# --------------------------
class LKACMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3, drop_prob=0.1):
        super(LKACMUNeXtBlock, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            spatial_mixer = LKASpatialBlock(dim=ch_in, k=k, drop_prob=drop_prob)

            # 优化点 5: 将 FFN 的通道膨胀率从 4 倍砍到 2 倍，减少参数量
            channel_mixer = nn.Sequential(
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_in * 2, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(drop_prob),  # FFN 中也加入正则化
                nn.Conv2d(ch_in * 2, ch_in, kernel_size=1),
                nn.Dropout2d(drop_prob)
            )

            nn.init.normal_(channel_mixer[4].weight, std=0.01)
            nn.init.constant_(channel_mixer[4].bias, 0)

            self.blocks.append(nn.ModuleDict({
                'spatial': spatial_mixer,
                'ffn': channel_mixer
            }))

        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        for blk in self.blocks:
            x = blk['spatial'](x)
            x = x + blk['ffn'](x)

        x = self.up(x)
        return x


# --------------------------
# 主网络: CMUNeXt_LKA (微调版)
# --------------------------
class CMUNeXt_LKA(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7], drop_prob=0.1):
        super(CMUNeXt_LKA, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])

        # 传入统一的 drop_prob
        self.encoder1 = LKACMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0],
                                        drop_prob=drop_prob)
        self.encoder2 = LKACMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1],
                                        drop_prob=drop_prob)
        self.encoder3 = LKACMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2],
                                        drop_prob=drop_prob)
        self.encoder4 = LKACMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3],
                                        drop_prob=drop_prob)
        self.encoder5 = LKACMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4],
                                        drop_prob=drop_prob)

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