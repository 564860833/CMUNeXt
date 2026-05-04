import torch
import torch.nn as nn
import torch.nn.functional as F


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
                    # depthwise large-kernel convolution
                    nn.Conv2d(
                        ch_in,
                        ch_in,
                        kernel_size=(k, k),
                        groups=ch_in,
                        padding=(k // 2, k // 2),
                        bias=True
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1), bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1), bias=True),
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
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class fusion_conv(nn.Module):
    """
    Original CMUNeXt fusion block, kept for baseline comparison.
    """
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        groups = 2 if ch_in % 2 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=groups, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1), bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1), bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class DWConvBNAct(nn.Module):
    """
    Lightweight depthwise convolution block.
    Used in SA-FSF to keep the added computation low.
    """
    def __init__(self, channels, kernel_size=3, act=True):
        super(DWConvBNAct, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)


class SpeckleAwareFrequencySkipFusion(nn.Module):
    """
    SA-FSF: Speckle-aware Frequency-guided Skip Fusion.

    This module replaces the original "concat + fusion_conv" skip fusion in CMUNeXt.

    Args:
        skip_channels: channels of encoder skip feature.
        dec_channels:  channels of upsampled decoder feature.
        out_channels:  output channels after fusion.
        low_kernel:    kernel size for low-frequency structural extraction.
        reduction:     channel reduction ratio used in frequency gates.

    Forward:
        x_skip: encoder feature, e.g. x4/x3/x2/x1.
        x_dec:  upsampled decoder feature, e.g. Up5(x5).
    """
    def __init__(
        self,
        skip_channels,
        dec_channels,
        out_channels,
        low_kernel=7,
        reduction=4
    ):
        super(SpeckleAwareFrequencySkipFusion, self).__init__()

        hidden_channels = max(skip_channels // reduction, 4)
        fuse_channels = skip_channels + dec_channels
        fuse_groups = 2 if fuse_channels % 2 == 0 else 1

        # Align decoder semantics to skip-channel space.
        self.dec_proj = nn.Sequential(
            nn.Conv2d(dec_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.GELU()
        )

        # Low-frequency branch:
        # depthwise large-kernel filtering captures smooth lesion structure
        # and suppresses random high-frequency speckle responses.
        self.low_pass = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                skip_channels,
                kernel_size=low_kernel,
                stride=1,
                padding=low_kernel // 2,
                groups=skip_channels,
                bias=False
            ),
            nn.BatchNorm2d(skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels)
        )

        # High-frequency branch:
        # x_high = x_skip - x_low.
        # This branch denoises high-frequency residuals before gating them.
        self.high_denoise = nn.Sequential(
            DWConvBNAct(skip_channels, kernel_size=3, act=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.GELU()
        )

        # High-frequency gate:
        # decoder semantics guide the module to preserve lesion-boundary high frequency
        # and suppress speckle-like high frequency.
        self.high_gate = nn.Sequential(
            nn.Conv2d(skip_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, skip_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Low-frequency gate:
        # selects useful smooth structural components of the lesion/background.
        self.low_gate = nn.Sequential(
            nn.Conv2d(skip_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, skip_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Final fusion:
        # keep the spirit of CMUNeXt fusion_conv, but use refined skip feature.
        self.fuse = nn.Sequential(
            nn.Conv2d(
                fuse_channels,
                fuse_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=fuse_groups,
                bias=True
            ),
            nn.GELU(),
            nn.BatchNorm2d(fuse_channels),
            nn.Conv2d(fuse_channels, out_channels * 4, kernel_size=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(out_channels * 4),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x_skip, x_dec):
        if x_dec.shape[-2:] != x_skip.shape[-2:]:
            x_dec = F.interpolate(
                x_dec,
                size=x_skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x_dec_sem = self.dec_proj(x_dec)

        # High-low decomposition.
        x_low = self.low_pass(x_skip)
        x_high = x_skip - x_low

        # Denoise high-frequency residuals.
        x_high = self.high_denoise(x_high)

        # Semantic-conditioned frequency gates.
        high_gate = self.high_gate(torch.cat([x_high, x_dec_sem], dim=1))
        low_gate = self.low_gate(torch.cat([x_low, x_dec_sem], dim=1))

        # Refine skip feature.
        # Residual form keeps the original skip information stable.
        x_skip_refined = x_skip + high_gate * x_high + low_gate * x_low

        # Fuse refined skip feature with decoder feature.
        x_out = self.fuse(torch.cat([x_skip_refined, x_dec], dim=1))
        return x_out


class CMUNeXt(nn.Module):
    """
    Original CMUNeXt architecture.
    Kept unchanged for fair baseline comparison.
    """
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=[16, 32, 128, 160, 256],
        depths=[1, 1, 1, 3, 1],
        kernels=[3, 3, 7, 7, 7]
    ):
        super(CMUNeXt, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
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


class CMUNeXt_SAFSF(nn.Module):
    """
    CMUNeXt with SA-FSF.

    Main difference from original CMUNeXt:
        Original:
            d = torch.cat((x_skip, d), dim=1)
            d = fusion_conv(d)

        Proposed:
            d = SpeckleAwareFrequencySkipFusion(x_skip, d)

    This keeps the encoder and upsampling blocks unchanged and only replaces
    the decoder skip-fusion blocks.
    """
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=[16, 32, 128, 160, 256],
        depths=[1, 1, 1, 3, 1],
        kernels=[3, 3, 7, 7, 7],
        low_kernel=7,
        reduction=4
    ):
        super(CMUNeXt_SAFSF, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder upsampling
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])

        # Decoder fusion.
        # Keep deeper skips as original CMUNeXt fusion and apply SA-FSF only on
        # shallow skips (x2, x1), where BUSI boundary details are most critical.
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up_conv3 = SpeckleAwareFrequencySkipFusion(
            skip_channels=dims[1],
            dec_channels=dims[1],
            out_channels=dims[1],
            low_kernel=low_kernel,
            reduction=reduction
        )
        self.Up_conv2 = SpeckleAwareFrequencySkipFusion(
            skip_channels=dims[0],
            dec_channels=dims[0],
            out_channels=dims[0],
            low_kernel=low_kernel,
            reduction=reduction
        )

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
        d3 = self.Up_conv3(x2, d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(x1, d2)

        d1 = self.Conv_1x1(d2)
        return d1


def cmunext(
    input_channel=3,
    num_classes=1,
    dims=[16, 32, 128, 160, 256],
    depths=[1, 1, 1, 3, 1],
    kernels=[3, 3, 7, 7, 7]
):
    return CMUNeXt(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes
    )


def cmunext_s(
    input_channel=3,
    num_classes=1,
    dims=[8, 16, 32, 64, 128],
    depths=[1, 1, 1, 1, 1],
    kernels=[3, 3, 7, 7, 9]
):
    return CMUNeXt(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes
    )


def cmunext_l(
    input_channel=3,
    num_classes=1,
    dims=[32, 64, 128, 256, 512],
    depths=[1, 1, 1, 6, 3],
    kernels=[3, 3, 7, 7, 7]
):
    return CMUNeXt(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes
    )


def cmunext_safsf(
    input_channel=3,
    num_classes=1,
    dims=[16, 32, 128, 160, 256],
    depths=[1, 1, 1, 3, 1],
    kernels=[3, 3, 7, 7, 7],
    low_kernel=7,
    reduction=4
):
    return CMUNeXt_SAFSF(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes,
        low_kernel=low_kernel,
        reduction=reduction
    )


def cmunext_s_safsf(
    input_channel=3,
    num_classes=1,
    dims=[8, 16, 32, 64, 128],
    depths=[1, 1, 1, 1, 1],
    kernels=[3, 3, 7, 7, 9],
    low_kernel=7,
    reduction=4
):
    return CMUNeXt_SAFSF(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes,
        low_kernel=low_kernel,
        reduction=reduction
    )


def cmunext_l_safsf(
    input_channel=3,
    num_classes=1,
    dims=[32, 64, 128, 256, 512],
    depths=[1, 1, 1, 6, 3],
    kernels=[3, 3, 7, 7, 7],
    low_kernel=7,
    reduction=4
):
    return CMUNeXt_SAFSF(
        dims=dims,
        depths=depths,
        kernels=kernels,
        input_channel=input_channel,
        num_classes=num_classes,
        low_kernel=low_kernel,
        reduction=reduction
    )


if __name__ == "__main__":
    model = cmunext_s_safsf(input_channel=3, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
