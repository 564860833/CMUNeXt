import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_DualGAG import (
    CMUNeXtBlock,
    DualGatedAttentionGate,
    conv_block,
    fusion_conv,
    up_conv,
)
from src.network.conv_based.CMUNeXt_SpeckleEnhance import DDSR


class CMUNeXt_DualGAG_SpeckleEnhance(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        gag_stages=(2, 3),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    ):
        super().__init__()
        self.ddsr_stages = set(ddsr_stages)
        self.gag_stages = set(gag_stages)

        valid_stages = {0, 1, 2, 3}
        invalid_ddsr = self.ddsr_stages - valid_stages
        invalid_gag = self.gag_stages - valid_stages
        if invalid_ddsr:
            raise ValueError(f"Unsupported DDSR stages: {sorted(invalid_ddsr)}")
        if invalid_gag:
            raise ValueError(f"Unsupported DualGAG stages: {sorted(invalid_gag)}")

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        skip_dims = [dims[0], dims[1], dims[2], dims[3]]
        self.ddsr_modules = nn.ModuleDict()
        for stage in sorted(self.ddsr_stages):
            self.ddsr_modules[str(stage)] = DDSR(
                channels=skip_dims[stage],
                smooth_k=ddsr_smooth_k,
                alpha_init_raw=alpha_init_raw,
            )

        gag_specs = {
            3: (dims[3], dims[3], max(8, dims[3] // 2), 4),
            2: (dims[2], dims[2], max(8, dims[2] // 2), 4),
            1: (dims[1], dims[1], max(8, dims[1] // 2), 4),
            0: (dims[0], dims[0], max(8, dims[0] // 2), 2),
        }
        self.gag_modules = nn.ModuleDict()
        for stage in sorted(self.gag_stages):
            f_g, f_l, f_int, groups = gag_specs[stage]
            self.gag_modules[str(stage)] = DualGatedAttentionGate(
                F_g=f_g,
                F_l=f_l,
                F_int=f_int,
                groups=groups,
            )

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
        key = str(stage)
        if key not in self.ddsr_modules:
            return x
        return self.ddsr_modules[key](x)

    def _apply_gag(self, g, x, stage):
        key = str(stage)
        if key not in self.gag_modules:
            return x
        return self.gag_modules[key](g=g, x=x)

    def forward(self, x):
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


def cmunext_dualgag_speckleenhance(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
):
    return CMUNeXt_DualGAG_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=ddsr_stages,
        gag_stages=gag_stages,
        ddsr_smooth_k=ddsr_smooth_k,
        alpha_init_raw=-5.3,
    )


def cmunext_dualgag_speckleenhance_full(input_channel=3, num_classes=1):
    return CMUNeXt_DualGAG_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1, 2, 3),
        gag_stages=(0, 1, 2, 3),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    )


def cmunext_dualgag_speckleenhance_s(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
):
    return CMUNeXt_DualGAG_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        ddsr_stages=ddsr_stages,
        gag_stages=gag_stages,
        ddsr_smooth_k=ddsr_smooth_k,
        alpha_init_raw=-5.3,
    )


def cmunext_dualgag_speckleenhance_l(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
):
    return CMUNeXt_DualGAG_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=ddsr_stages,
        gag_stages=gag_stages,
        ddsr_smooth_k=ddsr_smooth_k,
        alpha_init_raw=-5.3,
    )


CMUNeXt_SpeckleEnhance_DualGAG = CMUNeXt_DualGAG_SpeckleEnhance


def cmunext_speckle_dualgag(input_channel=3, num_classes=1):
    return cmunext_dualgag_speckleenhance(
        input_channel=input_channel,
        num_classes=num_classes,
    )
