import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_BUGR import BUGR
from src.network.conv_based.CMUNeXt_SpeckleEnhance import (
    CMUNeXtBlock,
    DDSR,
    conv_block,
    fusion_conv,
    up_conv,
)


class CMUNeXt_BUGR_SpeckleEnhance(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    ):
        super().__init__()
        self.ddsr_stages = set(ddsr_stages)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        skip_dims = [dims[0], dims[1], dims[2], dims[3]]
        self.ddsr_modules = nn.ModuleDict()
        for stage in ddsr_stages:
            self.ddsr_modules[str(stage)] = DDSR(
                channels=skip_dims[stage],
                smooth_k=ddsr_smooth_k,
                alpha_init_raw=alpha_init_raw,
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
        self.bugr = BUGR(ch_in=dims[0], num_classes=num_classes)

    def _apply_ddsr(self, x, stage):
        if stage in self.ddsr_stages:
            return self.ddsr_modules[str(stage)](x)
        return x

    def forward(self, x, return_aux=None):
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
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        pred_main = self.Conv_1x1(d2)
        pred_refined, boundary_map, uncertainty_map = self.bugr(d2, pred_main)

        if return_aux is None:
            return_aux = self.training

        if return_aux:
            return {
                "seg": pred_refined,
                "pred_main": pred_main,
                "pred_refined": pred_refined,
                "boundary_map": boundary_map,
                "uncertainty_map": uncertainty_map,
            }
        return pred_refined


def cmunext_bugr_speckleenhance(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    )


def cmunext_bugr_speckleenhance_s(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        ddsr_stages=(0, 1),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    )


def cmunext_bugr_speckleenhance_l(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR_SpeckleEnhance(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        ddsr_stages=(0, 1),
        ddsr_smooth_k=5,
        alpha_init_raw=-5.3,
    )
