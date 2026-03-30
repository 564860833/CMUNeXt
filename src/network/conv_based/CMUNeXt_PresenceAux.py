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


class LesionExistHead(nn.Module):
    def __init__(self, in_channels, hidden_ratio=4, dropout=0.1):
        super().__init__()
        hidden = max(16, in_channels // hidden_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        return self.fc(feat)


class CMUNeXt_PresenceAux(nn.Module):
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

        self.exist_head = LesionExistHead(dims[4])

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

        cls_logit = self.exist_head(x5)

        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        seg_logit = self.Conv_1x1(d2)

        if not return_aux:
            return seg_logit

        return {
            "seg": seg_logit,
            "cls": cls_logit,
        }


def dice_loss_with_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


@torch.no_grad()
def build_presence_target(mask):
    mask = mask.float()
    return (mask.flatten(1).sum(dim=1, keepdim=True) > 0).float()


class PresenceAuxLoss(nn.Module):
    def __init__(self, cls_weight=0.2, seg_bce_weight=1.0, seg_dice_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight

    def forward(self, outputs, mask):
        seg_logit = outputs["seg"]
        cls_logit = outputs["cls"]
        mask = mask.float()
        cls_target = build_presence_target(mask)

        seg_bce = F.binary_cross_entropy_with_logits(seg_logit, mask)
        seg_dice = dice_loss_with_logits(seg_logit, mask)
        seg_loss = self.seg_bce_weight * seg_bce + self.seg_dice_weight * seg_dice

        cls_loss = F.binary_cross_entropy_with_logits(cls_logit, cls_target)
        total = seg_loss + self.cls_weight * cls_loss
        return total, {
            "loss_total": total.detach(),
            "loss_seg": seg_loss.detach(),
            "loss_cls": cls_loss.detach(),
        }


@torch.no_grad()
def predict_with_presence_gate(outputs, cls_threshold=0.35, hard_gate=True):
    seg_prob = torch.sigmoid(outputs["seg"])
    cls_prob = torch.sigmoid(outputs["cls"]).view(-1, 1, 1, 1)
    if hard_gate:
        keep = (cls_prob >= cls_threshold).float()
        seg_prob = seg_prob * keep
    else:
        seg_prob = seg_prob * cls_prob
    return seg_prob, cls_prob.flatten(1)


def cmunext_presenceaux(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
):
    return CMUNeXt_PresenceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
    )


def cmunext_presenceaux_s(input_channel=3, num_classes=1):
    return CMUNeXt_PresenceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
    )


def cmunext_presenceaux_l(input_channel=3, num_classes=1):
    return CMUNeXt_PresenceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
    )
