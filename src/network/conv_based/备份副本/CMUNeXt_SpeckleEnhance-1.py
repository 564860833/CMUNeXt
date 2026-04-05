import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _make_norm(num_channels: int, norm: str = "gn", num_groups: int = 8):
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)

    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        bias: bool = False,
        norm: str = "gn",
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            _make_norm(out_ch, norm=norm),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpeckleAwareEnhancer(nn.Module):
    """
    Multi-scale speckle-aware enhancement with softer, 1-centered gates so
    the module can preserve boundaries instead of only suppressing responses.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        norm: str = "gn",
        use_residual_scale: bool = True,
    ):
        super().__init__()
        hidden = max(8, channels // reduction)

        self.smooth3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.smooth_fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, norm=norm, act=True),
            nn.Conv2d(channels, 2, kernel_size=1, bias=True),
        )

        self.low_proj = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1, norm=norm, act=True),
            ConvNormAct(channels, channels, kernel_size=3, norm=norm, act=True),
        )
        self.high_proj = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3, groups=channels, norm=norm, act=True),
            ConvNormAct(channels, channels, kernel_size=1, norm=norm, act=True),
            ConvNormAct(channels, channels, kernel_size=3, groups=channels, norm=norm, act=True),
        )

        gate_in_ch = channels * 4
        self.channel_gate = nn.Sequential(
            nn.Conv2d(gate_in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
        )
        self.channel_scale = nn.Parameter(torch.tensor(0.1))

        spatial_groups = max(1, channels // 8) if channels >= 8 else 1
        while spatial_groups > 1 and channels % spatial_groups != 0:
            spatial_groups -= 1
        self.spatial_refine = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=3, norm=norm, act=True),
            ConvNormAct(channels, channels, kernel_size=3, groups=spatial_groups, norm=norm, act=True),
        )
        self.spatial_gate = nn.Conv2d(channels, 2, kernel_size=1, bias=True)
        self.spatial_scale = nn.Parameter(torch.tensor([0.1, 0.1]))

        self.fuse = nn.Sequential(
            ConvNormAct(channels * 3, channels, kernel_size=3, norm=norm, act=True),
            ConvNormAct(channels, channels, kernel_size=3, norm=norm, act=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_norm(channels, norm=norm),
        )

        self.use_residual_scale = use_residual_scale
        if use_residual_scale:
            self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.out_act = nn.GELU()

    @staticmethod
    def _centered_gate(raw_gate: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(raw_gate)
        gate = 2.0 * gate - 1.0
        gate = 1.0 + torch.tanh(scale) * gate
        return gate

    def forward(self, x):
        smooth3 = self.smooth3(x)
        smooth5 = self.smooth5(x)

        smooth_cat = torch.cat([smooth3, smooth5], dim=1)
        smooth_logits = self.smooth_fuse(smooth_cat)
        smooth_weights = torch.softmax(smooth_logits, dim=1)
        smooth = smooth_weights[:, 0:1] * smooth3 + smooth_weights[:, 1:2] * smooth5

        residual = x - smooth
        low = self.low_proj(smooth)
        high = self.high_proj(residual)

        low_avg = F.adaptive_avg_pool2d(low, 1)
        low_max = F.adaptive_max_pool2d(low, 1)
        high_avg = F.adaptive_avg_pool2d(high, 1)
        high_max = F.adaptive_max_pool2d(high, 1)
        ch_stat = torch.cat([low_avg, low_max, high_avg, high_max], dim=1)

        ch_gate = self.channel_gate(ch_stat)
        low_ch_gate, high_ch_gate = torch.chunk(ch_gate, 2, dim=1)
        low_ch_gate = self._centered_gate(low_ch_gate, self.channel_scale)
        high_ch_gate = self._centered_gate(high_ch_gate, self.channel_scale)

        spatial_feat = self.spatial_refine(torch.cat([low, high], dim=1))
        sp_gate = self.spatial_gate(spatial_feat)
        low_sp_gate = self._centered_gate(sp_gate[:, 0:1], self.spatial_scale[0])
        high_sp_gate = self._centered_gate(sp_gate[:, 1:2], self.spatial_scale[1])

        low = low * low_ch_gate * low_sp_gate
        high = high * high_ch_gate * high_sp_gate

        fused = self.fuse(torch.cat([x, low, high], dim=1))
        if self.use_residual_scale:
            out = x + torch.tanh(self.res_scale) * fused
        else:
            out = x + fused
        return self.out_act(out)


class CMUNeXt_SpeckleEnhance(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        speckle_reduction=4,
        speckle_norm="gn",
    ):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.speckle1 = SpeckleAwareEnhancer(
            channels=dims[0], reduction=speckle_reduction, norm=speckle_norm
        )

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
        x1 = self.speckle1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        return self.Conv_1x1(d2)


def cmunext_speckle(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    speckle_reduction=4,
    speckle_norm="gn",
):
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        speckle_reduction=speckle_reduction,
        speckle_norm=speckle_norm,
    )


def cmunext_speckle_s(
    input_channel=3,
    num_classes=1,
    speckle_reduction=4,
    speckle_norm="gn",
):
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        speckle_reduction=speckle_reduction,
        speckle_norm=speckle_norm,
    )


def cmunext_speckle_l(
    input_channel=3,
    num_classes=1,
    speckle_reduction=4,
    speckle_norm="gn",
):
    return CMUNeXt_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        speckle_reduction=speckle_reduction,
        speckle_norm=speckle_norm,
    )
