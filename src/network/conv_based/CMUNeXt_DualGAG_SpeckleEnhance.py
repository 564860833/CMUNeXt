import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_DualGAG import (
    CMUNeXtBlock,
    _make_gag,
    _normalize_gag_stages,
    conv_block,
    fusion_conv,
    up_conv,
)
from src.network.conv_based.CMUNeXt_SpeckleEnhance import DDSR


def _normalize_ddsr_stages(ddsr_stages):
    if isinstance(ddsr_stages, str):
        ddsr_stages = ddsr_stages.split(",")

    stages = []
    for stage in ddsr_stages:
        stage = int(stage)
        if stage not in {0, 1, 2, 3}:
            raise ValueError(f"Unsupported DDSR stage: {stage}")
        if stage not in stages:
            stages.append(stage)

    if not stages:
        raise ValueError("DDSR stages must include at least one stage.")
    return tuple(stages)


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
        ddsr_max_scale=0.05,
        ddsr_skip_only=True,
        ddsr_aux_init=0.1,
        alpha_init_raw=-5.3,
    ):
        super().__init__()
        self.ddsr_stages = set(_normalize_ddsr_stages(ddsr_stages))
        self.gag_stages = set(_normalize_gag_stages(gag_stages))
        self.ddsr_skip_only = ddsr_skip_only

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
                max_scale=ddsr_max_scale,
            )

        self.ddsr_aux_scales = nn.ParameterDict()
        for stage in sorted(self.ddsr_stages):
            self.ddsr_aux_scales[str(stage)] = nn.Parameter(torch.tensor(float(ddsr_aux_init)))

        self.gag_modules = nn.ModuleDict()
        for stage in sorted(self.gag_stages):
            self.gag_modules[str(stage)] = _make_gag(stage, dims)

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

    def _merge_skip(self, gated_raw_skip, raw_skip, ddsr_skip, stage):
        key = str(stage)
        if key not in self.ddsr_aux_scales:
            return gated_raw_skip
        scale = torch.tanh(self.ddsr_aux_scales[key])
        return gated_raw_skip + scale * (ddsr_skip - raw_skip)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        s1 = self._apply_ddsr(x1, 0)
        x1_next = x1 if self.ddsr_skip_only else s1

        x2 = self.Maxpool(x1_next)
        x2 = self.encoder2(x2)
        s2 = self._apply_ddsr(x2, 1)
        x2_next = x2 if self.ddsr_skip_only else s2

        x3 = self.Maxpool(x2_next)
        x3 = self.encoder3(x3)
        s3 = self._apply_ddsr(x3, 2)
        x3_next = x3 if self.ddsr_skip_only else s3

        x4 = self.Maxpool(x3_next)
        x4 = self.encoder4(x4)
        s4 = self._apply_ddsr(x4, 3)
        x4_next = x4 if self.ddsr_skip_only else s4

        x5 = self.Maxpool(x4_next)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        x4_p = self._apply_gag(d5, x4, 3)
        x4_p = self._merge_skip(x4_p, x4, s4, 3)
        d5 = self.Up_conv5(torch.cat((x4_p, d5), dim=1))

        d4 = self.Up4(d5)
        x3_p = self._apply_gag(d4, x3, 2)
        x3_p = self._merge_skip(x3_p, x3, s3, 2)
        d4 = self.Up_conv4(torch.cat((x3_p, d4), dim=1))

        d3 = self.Up3(d4)
        x2_p = self._apply_gag(d3, x2, 1)
        x2_p = self._merge_skip(x2_p, x2, s2, 1)
        d3 = self.Up_conv3(torch.cat((x2_p, d3), dim=1))

        d2 = self.Up2(d3)
        x1_p = self._apply_gag(d2, x1, 0)
        x1_p = self._merge_skip(x1_p, x1, s1, 0)
        d2 = self.Up_conv2(torch.cat((x1_p, d2), dim=1))

        return self.Conv_1x1(d2)


def cmunext_dualgag_speckleenhance(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
    ddsr_max_scale=0.05,
    ddsr_skip_only=True,
    ddsr_aux_init=0.1,
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
        ddsr_max_scale=ddsr_max_scale,
        ddsr_skip_only=ddsr_skip_only,
        ddsr_aux_init=ddsr_aux_init,
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
        ddsr_max_scale=0.05,
        ddsr_skip_only=True,
        ddsr_aux_init=0.1,
        alpha_init_raw=-5.3,
    )


def cmunext_dualgag_speckleenhance_s(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
    ddsr_max_scale=0.05,
    ddsr_skip_only=True,
    ddsr_aux_init=0.1,
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
        ddsr_max_scale=ddsr_max_scale,
        ddsr_skip_only=ddsr_skip_only,
        ddsr_aux_init=ddsr_aux_init,
        alpha_init_raw=-5.3,
    )


def cmunext_dualgag_speckleenhance_l(
    input_channel=3,
    num_classes=1,
    ddsr_stages=(0, 1),
    gag_stages=(2, 3),
    ddsr_smooth_k=5,
    ddsr_max_scale=0.05,
    ddsr_skip_only=True,
    ddsr_aux_init=0.1,
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
        ddsr_max_scale=ddsr_max_scale,
        ddsr_skip_only=ddsr_skip_only,
        ddsr_aux_init=ddsr_aux_init,
        alpha_init_raw=-5.3,
    )


CMUNeXt_SpeckleEnhance_DualGAG = CMUNeXt_DualGAG_SpeckleEnhance


def cmunext_speckle_dualgag(input_channel=3, num_classes=1):
    return cmunext_dualgag_speckleenhance(
        input_channel=input_channel,
        num_classes=num_classes,
    )
