import numpy as np
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
            nn.Upsample(scale_factor=2, mode="bilinear"),
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


class DistanceHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(16, in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.head(x)


class CMUNeXt_DistanceAux(nn.Module):
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

        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.distance_head = DistanceHead(dims[0])

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
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        seg_logit = self.seg_head(d2)

        if not return_aux:
            return seg_logit

        distance_logit = self.distance_head(d2)
        return {
            "seg": seg_logit,
            "dist": distance_logit,
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
def build_signed_distance_target(mask, max_distance=32.0):
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError("CMUNeXt_DistanceAux requires scipy for distance-transform targets.") from exc

    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    target = np.zeros_like(mask_np, dtype=np.float32)

    for batch_idx in range(mask_np.shape[0]):
        foreground = mask_np[batch_idx, 0].astype(bool)
        if not foreground.any():
            continue

        pos_dist = distance_transform_edt(foreground)
        neg_dist = distance_transform_edt(~foreground)
        signed_distance = (neg_dist - pos_dist) / float(max_distance)
        target[batch_idx, 0] = np.clip(signed_distance, -1.0, 1.0)

    return torch.from_numpy(target).to(device=mask.device, dtype=mask.dtype)


class DistanceAuxLoss(nn.Module):
    def __init__(
        self,
        dist_weight=0.15,
        seg_bce_weight=1.0,
        seg_dice_weight=1.0,
        dist_band_weight=2.0,
        dist_band_scale=4.0,
        max_distance=32.0,
    ):
        super().__init__()
        self.dist_weight = dist_weight
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.dist_band_weight = dist_band_weight
        self.dist_band_scale = dist_band_scale
        self.max_distance = max_distance

    def forward(self, outputs, mask, distance_target=None, dist_weight=None):
        seg_logit = outputs["seg"]
        dist_logit = outputs["dist"]
        mask = mask.float()
        effective_dist_weight = self.dist_weight if dist_weight is None else dist_weight

        seg_bce = F.binary_cross_entropy_with_logits(seg_logit, mask)
        seg_dice = dice_loss_with_logits(seg_logit, mask)
        seg_loss = self.seg_bce_weight * seg_bce + self.seg_dice_weight * seg_dice

        valid_samples = (mask.flatten(1).sum(dim=1) > 0)
        if valid_samples.any():
            if distance_target is None:
                dist_target_valid = build_signed_distance_target(mask[valid_samples], max_distance=self.max_distance)
            else:
                dist_target_valid = distance_target[valid_samples].to(device=mask.device, dtype=mask.dtype)
            dist_pred = torch.tanh(dist_logit[valid_samples])
            dist_error = F.smooth_l1_loss(dist_pred, dist_target_valid, reduction="none")
            band_weight = 1.0 + self.dist_band_weight * torch.exp(
                -self.dist_band_scale * torch.abs(dist_target_valid)
            )
            dist_loss = (dist_error * band_weight).mean()
        else:
            dist_loss = seg_logit.new_tensor(0.0)

        total = seg_loss + effective_dist_weight * dist_loss
        return total, {
            "loss_total": total.detach(),
            "loss_seg": seg_loss.detach(),
            "loss_dist": dist_loss.detach(),
        }


def cmunext_distanceaux(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
):
    return CMUNeXt_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
    )


def cmunext_distanceaux_s(input_channel=3, num_classes=1):
    return CMUNeXt_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
    )


def cmunext_distanceaux_l(input_channel=3, num_classes=1):
    return CMUNeXt_DistanceAux(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
    )
