import torch
import torch.nn as nn


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
            ) for _ in range(depth)]
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


class SelectiveResidualGroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, groups=4):
        super(SelectiveResidualGroupedAttentionGate, self).__init__()
        actual_groups = groups if (F_int % groups == 0 and F_l % groups == 0) else 1
        self.groups = actual_groups

        # Use stable semantic alignment first, then retain grouped local modeling
        # as the task-specific innovation for BUSI-style noisy boundaries.
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
            nn.GELU()
        )
        self.psi = nn.Conv2d(F_int, actual_groups, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate_scale = nn.Parameter(torch.full((actual_groups,), 0.1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        fused = self.relu(self.W_g(g) + self.W_x(x))
        fused = self.refine(fused)

        # Center the gate around zero and apply it residually so the baseline
        # skip path remains intact unless a learned modulation is helpful.
        gate = torch.sigmoid(self.psi(fused))
        gate = 2.0 * gate - 1.0
        gate = 1.0 + torch.tanh(self.gate_scale).view(1, self.groups, 1, 1) * gate

        if self.groups == 1:
            return x * gate

        gated_groups = []
        for group_idx, chunk in enumerate(torch.chunk(x, self.groups, dim=1)):
            gated_groups.append(chunk * gate[:, group_idx:group_idx + 1])
        return torch.cat(gated_groups, dim=1)


class CMUNeXt_GAG(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GAG, self).__init__()
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

        # Gate only deep skips where semantic guidance is strong enough.
        self.gag5 = SelectiveResidualGroupedAttentionGate(F_g=dims[3], F_l=dims[3], F_int=dims[3] // 2, groups=4)
        self.gag4 = SelectiveResidualGroupedAttentionGate(F_g=dims[2], F_l=dims[2], F_int=dims[2] // 2, groups=4)

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
        x4_refined = self.gag5(g=d5, x=x4)
        d5 = torch.cat((x4_refined, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_refined = self.gag4(g=d4, x=x3)
        d4 = torch.cat((x3_refined, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # Keep shallow skips unchanged to preserve fine BUSI boundary detail.
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


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
    model = cmunext_gag(input_channel=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("CMUNeXt-GAG Output shape:", output.shape)
