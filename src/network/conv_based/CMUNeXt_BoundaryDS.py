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


class BoundaryHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(16, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x, out_size):
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear")


class CMUNeXt_BoundaryDS(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        use_boundary_all_scales=False,
    ):
        super().__init__()
        self.use_boundary_all_scales = use_boundary_all_scales

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

        self.edge_head2 = BoundaryHead(dims[0])
        self.edge_head3 = BoundaryHead(dims[1])
        self.edge_head4 = BoundaryHead(dims[2])
        self.edge_head5 = BoundaryHead(dims[3])

    def forward(self, x, return_aux=True):
        out_size = x.shape[-2:]

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

        seg_logit = self.Conv_1x1(d2)

        if not return_aux:
            return seg_logit

        edge_logits = {
            "edge2": self.edge_head2(d2, out_size),
            "edge3": self.edge_head3(d3, out_size),
        }
        if self.use_boundary_all_scales:
            edge_logits["edge4"] = self.edge_head4(d4, out_size)
            edge_logits["edge5"] = self.edge_head5(d5, out_size)

        return {
            "seg": seg_logit,
            "edges": edge_logits,
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
def build_boundary_target(mask, kernel_size=3):
    # Match the thin inner-boundary definition used by boundary_scores() so
    # the auxiliary supervision optimizes for the same contour notion that the
    # validation metrics later measure.
    mask = (mask > 0.5).float()
    pad = kernel_size // 2
    ero = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    edge = (mask - ero) > 0
    return edge.float()


class BoundaryDeepSupervisionLoss(nn.Module):
    def __init__(self, edge_weight=0.2, seg_bce_weight=1.0, seg_dice_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.edge_loss_weights = {
            "edge2": 1.0,
            "edge3": 0.5,
            "edge4": 0.25,
            "edge5": 0.25,
        }

    def forward(self, outputs, mask):
        seg_logit = outputs["seg"]
        edge_logits = outputs["edges"]
        mask = mask.float()
        edge_target = build_boundary_target(mask)
        valid_edge_samples = (mask.flatten(1).sum(dim=1) > 0)

        seg_bce = F.binary_cross_entropy_with_logits(seg_logit, mask)
        seg_dice = dice_loss_with_logits(seg_logit, mask)
        seg_loss = self.seg_bce_weight * seg_bce + self.seg_dice_weight * seg_dice

        weighted_edge_loss = seg_logit.new_tensor(0.0)
        total_edge_weight = 0.0
        if valid_edge_samples.any():
            edge_target_valid = edge_target[valid_edge_samples]
            for edge_name, edge_logit in edge_logits.items():
                edge_weight = self.edge_loss_weights.get(edge_name, 1.0)
                edge_logit_valid = edge_logit[valid_edge_samples]
                edge_bce = F.binary_cross_entropy_with_logits(edge_logit_valid, edge_target_valid)
                edge_dice = dice_loss_with_logits(edge_logit_valid, edge_target_valid)
                weighted_edge_loss = weighted_edge_loss + edge_weight * (edge_bce + edge_dice)
                total_edge_weight += edge_weight

        if total_edge_weight > 0:
            edge_loss = weighted_edge_loss / total_edge_weight
        else:
            edge_loss = seg_logit.new_tensor(0.0)

        total = seg_loss + self.edge_weight * edge_loss
        return total, {
            "loss_total": total.detach(),
            "loss_seg": seg_loss.detach(),
            "loss_edge": edge_loss.detach(),
        }


def cmunext_boundaryds(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    use_boundary_all_scales=False,
):
    return CMUNeXt_BoundaryDS(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        use_boundary_all_scales=use_boundary_all_scales,
    )


def cmunext_boundaryds_s(input_channel=3, num_classes=1, use_boundary_all_scales=False):
    return CMUNeXt_BoundaryDS(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        use_boundary_all_scales=use_boundary_all_scales,
    )


def cmunext_boundaryds_l(input_channel=3, num_classes=1, use_boundary_all_scales=False):
    return CMUNeXt_BoundaryDS(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        use_boundary_all_scales=use_boundary_all_scales,
    )
