import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_DistanceAux import DistanceHead
from src.network.conv_based.CMUNeXt_DualGAG import (
    CMUNeXtBlock,
    _GAG_STAGE_ATTRS,
    _make_gag,
    _normalize_gag_stages,
    conv_block,
    fusion_conv,
    up_conv,
)


class CMUNeXt_DualGAG_DistanceAux(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        gag_stages=None,
        use_shallow_gates=False,
    ):
        super().__init__()
        self.gag_stages = set(_normalize_gag_stages(gag_stages, use_shallow_gates))
        self.use_shallow_gates = bool(self.gag_stages & {0, 1})
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

        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.distance_head = DistanceHead(dims[0])

        for stage in sorted(self.gag_stages, reverse=True):
            setattr(self, _GAG_STAGE_ATTRS[stage], _make_gag(stage, dims))

    def _apply_gag(self, g, x, stage):
        if stage not in self.gag_stages:
            return x
        return getattr(self, _GAG_STAGE_ATTRS[stage])(g=g, x=x)

    def forward(self, x, return_aux=True):
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
        x4_p = self._apply_gag(g=d5, x=x4, stage=3)
        d5 = self.Up_conv5(torch.cat((x4_p, d5), dim=1))

        d4 = self.Up4(d5)
        x3_p = self._apply_gag(g=d4, x=x3, stage=2)
        d4 = self.Up_conv4(torch.cat((x3_p, d4), dim=1))

        d3 = self.Up3(d4)
        x2_p = self._apply_gag(g=d3, x=x2, stage=1)
        d3 = self.Up_conv3(torch.cat((x2_p, d3), dim=1))

        d2 = self.Up2(d3)
        x1_p = self._apply_gag(g=d2, x=x1, stage=0)
        d2 = self.Up_conv2(torch.cat((x1_p, d2), dim=1))

        seg_logit = self.seg_head(d2)
        if not return_aux:
            return seg_logit

        distance_logit = self.distance_head(d2)
        return {
            "seg": seg_logit,
            "dist": distance_logit,
        }


def cmunext_dualgag_distanceaux(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    gag_stages=(2, 3),
):
    return CMUNeXt_DualGAG_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        gag_stages=gag_stages,
    )


def cmunext_dualgag_distanceaux_s(input_channel=3, num_classes=1, gag_stages=(2, 3)):
    return CMUNeXt_DualGAG_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        gag_stages=gag_stages,
    )


def cmunext_dualgag_distanceaux_l(input_channel=3, num_classes=1, gag_stages=(2, 3)):
    return CMUNeXt_DualGAG_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 160, 224, 320),
        depths=(1, 1, 1, 3, 2),
        kernels=(3, 5, 7, 7, 9),
        gag_stages=gag_stages,
    )
