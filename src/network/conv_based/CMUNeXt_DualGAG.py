import torch
import torch.nn as nn


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
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
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
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                ch_in,
                                ch_in,
                                kernel_size=k,
                                groups=ch_in,
                                padding=k // 2,
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class DualGatedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, groups=4, reduction=8):
        super().__init__()
        actual_groups = groups if F_int % groups == 0 else 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.spatial_gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        channel_in = F_g + F_l
        channel_mid = max(8, channel_in // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channel_in, channel_mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_mid, F_l, kernel_size=1, bias=False),
        )
        self.channel_act = nn.Sigmoid()

    def forward(self, g, x):
        spatial_mask = self.spatial_gate(self.W_g(g) + self.W_x(x))
        joint = torch.cat([g, x], dim=1)
        channel_mask = self.channel_act(
            self.channel_mlp(self.avg_pool(joint)) + self.channel_mlp(self.max_pool(joint))
        )
        return x * spatial_mask * channel_mask + x


class CMUNeXt_DualGAG(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
    ):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.gag5 = DualGatedAttentionGate(F_g=dims[3], F_l=dims[3], F_int=max(8, dims[3] // 2), groups=4)
        self.gag4 = DualGatedAttentionGate(F_g=dims[2], F_l=dims[2], F_int=max(8, dims[2] // 2), groups=4)
        self.gag3 = DualGatedAttentionGate(F_g=dims[1], F_l=dims[1], F_int=max(8, dims[1] // 2), groups=4)
        self.gag2 = DualGatedAttentionGate(F_g=dims[0], F_l=dims[0], F_int=max(8, dims[0] // 2), groups=2)

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
        x4_p = self.gag5(g=d5, x=x4)
        d5 = self.Up_conv5(torch.cat((x4_p, d5), dim=1))

        d4 = self.Up4(d5)
        x3_p = self.gag4(g=d4, x=x3)
        d4 = self.Up_conv4(torch.cat((x3_p, d4), dim=1))

        d3 = self.Up3(d4)
        x2_p = self.gag3(g=d3, x=x2)
        d3 = self.Up_conv3(torch.cat((x2_p, d3), dim=1))

        d2 = self.Up2(d3)
        x1_p = self.gag2(g=d2, x=x1)
        d2 = self.Up_conv2(torch.cat((x1_p, d2), dim=1))

        return self.Conv_1x1(d2)


def cmunext_dualgag(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
):
    return CMUNeXt_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
    )


def cmunext_dualgag_s(input_channel=3, num_classes=1):
    return CMUNeXt_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
    )


def cmunext_dualgag_l(input_channel=3, num_classes=1):
    return CMUNeXt_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
    )
