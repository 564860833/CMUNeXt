import torch
import torch.nn as nn
import torch.nn.functional as F


def build_inner_boundary_target(mask, kernel_size=3):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    mask = mask > 0.5
    original_dim = mask.dim()
    if original_dim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        mask = mask.unsqueeze(1)
    elif original_dim != 4:
        raise ValueError("mask must be 2D, 3D, or 4D.")

    flat_mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]).bool()
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        dtype=torch.float32,
        device=flat_mask.device,
    )
    eroded = F.conv2d(
        flat_mask.float(),
        kernel,
        padding=kernel_size // 2,
    ) == float(kernel_size * kernel_size)
    boundary = flat_mask & (~eroded)
    has_boundary = boundary.flatten(1).any(dim=1).view(-1, 1, 1, 1)
    boundary = torch.where(has_boundary, boundary, flat_mask).float()
    boundary = boundary.view_as(mask)

    if original_dim == 2:
        return boundary[0, 0]
    if original_dim == 3:
        return boundary.squeeze(1)
    return boundary


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
                            nn.Conv2d(ch_in, ch_in, kernel_size=k, groups=ch_in, padding=k // 2),
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


class DDSR(nn.Module):
    def __init__(self, channels, smooth_k=5, alpha_init_raw=-5.3):
        super().__init__()
        self.smooth = nn.AvgPool2d(
            kernel_size=smooth_k,
            stride=1,
            padding=smooth_k // 2,
            count_include_pad=False,
        )
        self.lin_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.log_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.noise_pred = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), alpha_init_raw))

    @staticmethod
    def signed_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, x, return_noise=False):
        struct = self.smooth(x)
        detail = x - struct
        lin_feat = self.lin_branch(detail)

        x_log = self.signed_log(x)
        struct_log = self.smooth(x_log)
        log_detail = x_log - struct_log
        log_feat = self.log_branch(log_detail)

        noise = self.noise_pred(torch.cat([lin_feat, log_feat], dim=1))
        scale = F.softplus(self.alpha)
        scaled_noise = scale * noise
        denoised = x - scaled_noise

        if not return_noise:
            return denoised

        noise_magnitude = scaled_noise.abs().mean(dim=1, keepdim=True)
        return denoised, noise_magnitude


class BUGR(nn.Module):
    def __init__(self, ch_in, num_classes=1, compress_ch=4, use_noise=False):
        super().__init__()
        self.use_noise = use_noise

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.boundary_branch = nn.Sequential(
            nn.Conv2d(ch_in + 1, ch_in // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 2, ch_in // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.feat_compress = nn.Sequential(
            nn.Conv2d(ch_in, compress_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(compress_ch),
            nn.ReLU(inplace=True),
        )

        attn_in_ch = compress_ch + 2 + (1 if use_noise else 0)
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(attn_in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 2, num_classes, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.refine_head[-1].weight, 0)
        nn.init.constant_(self.refine_head[-1].bias, 0)

    def _compute_uncertainty(self, pred):
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        entropy = -pred * torch.log(pred) - (1.0 - pred) * torch.log(1.0 - pred)
        return entropy / 0.6931

    def _compute_sobel_prior(self, pred):
        boundary_list = []
        for channel_idx in range(pred.shape[1]):
            channel_pred = pred[:, channel_idx:channel_idx + 1, :, :]
            grad_x = F.conv2d(channel_pred, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel_pred, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            boundary_list.append(grad_mag)

        sobel_out = torch.cat(boundary_list, dim=1).mean(dim=1, keepdim=True)
        max_val = sobel_out.flatten(2).max(dim=2)[0].reshape(pred.shape[0], 1, 1, 1) + 1e-8
        return sobel_out / max_val

    def forward(self, f_decoder, pred_main_logits, noise_intensity=None):
        pred_sigmoid = torch.sigmoid(pred_main_logits)
        uncertainty_map = self._compute_uncertainty(pred_sigmoid)

        sobel_prior = self._compute_sobel_prior(pred_sigmoid)
        boundary_input = torch.cat([f_decoder, sobel_prior], dim=1)
        boundary_map = self.boundary_branch(boundary_input)

        f_compressed = self.feat_compress(f_decoder)
        attn_parts = [f_compressed, uncertainty_map, boundary_map]

        if self.use_noise and noise_intensity is not None:
            if noise_intensity.shape[-2:] != f_decoder.shape[-2:]:
                noise_intensity = F.interpolate(
                    noise_intensity,
                    size=f_decoder.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            attn_parts.append(noise_intensity)

        attn_input = torch.cat(attn_parts, dim=1)
        attention = self.attention_fusion(attn_input)

        f_refined = f_decoder * (1.0 + attention)
        delta = self.refine_head(f_refined)
        pred_refined = pred_main_logits + delta
        return pred_refined, boundary_map, uncertainty_map


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def _dice_loss(self, logits, target):
        pred = torch.sigmoid(logits)
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = self._dice_loss(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class ProbBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def _dice_loss(self, probs, target):
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        probs_flat = probs.flatten(1)
        target_flat = target.flatten(1)
        intersection = (probs_flat * target_flat).sum(1)
        union = probs_flat.sum(1) + target_flat.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, probs, target):
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probs, target)
        dice = self._dice_loss(probs, target)
        return self.bce_weight * bce + self.dice_weight * dice


class BUGRLoss(nn.Module):
    def __init__(
        self,
        lambda1=1.0,
        lambda2=0.5,
        edge_kernel_size=3,
        bce_weight=0.5,
        dice_weight=0.5,
        boundary_bce_weight=0.5,
        boundary_dice_weight=0.5,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.edge_k = edge_kernel_size
        self.seg_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        self.bdy_loss = ProbBCEDiceLoss(
            bce_weight=boundary_bce_weight,
            dice_weight=boundary_dice_weight,
        )

    def _extract_gt_edge(self, gt_mask):
        return build_inner_boundary_target(gt_mask, kernel_size=self.edge_k).float()

    def forward(self, pred_main, pred_refined, boundary_map, gt_mask):
        gt_mask = gt_mask.float()
        loss_main = self.seg_loss(pred_main, gt_mask)
        loss_refine = self.seg_loss(pred_refined, gt_mask)
        gt_edge = self._extract_gt_edge(gt_mask)
        loss_boundary = self.bdy_loss(boundary_map, gt_edge)

        loss_total = loss_main + self.lambda1 * loss_refine + self.lambda2 * loss_boundary
        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_main": loss_main.item(),
            "loss_refine": loss_refine.item(),
            "loss_boundary": loss_boundary.item(),
        }
        return loss_total, loss_dict


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
        self.bugr = BUGR(ch_in=dims[0], num_classes=num_classes, use_noise=True)

    def _apply_ddsr(self, x, stage, return_noise=False):
        if stage in self.ddsr_stages:
            return self.ddsr_modules[str(stage)](x, return_noise=return_noise)
        if return_noise:
            return x, None
        return x

    def _merge_noise_maps(self, noise_maps, target_size):
        if len(noise_maps) == 0:
            return None

        aligned = []
        for noise_map in noise_maps:
            if noise_map.shape[-2:] != target_size:
                noise_map = F.interpolate(
                    noise_map,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(noise_map)

        return torch.stack(aligned, dim=0).mean(dim=0)

    def forward(self, x, return_aux=None):
        noise_maps = []

        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x1, n1 = self._apply_ddsr(x1, 0, return_noise=True)
        if n1 is not None:
            noise_maps.append(n1)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x2, n2 = self._apply_ddsr(x2, 1, return_noise=True)
        if n2 is not None:
            noise_maps.append(n2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3, n3 = self._apply_ddsr(x3, 2, return_noise=True)
        if n3 is not None:
            noise_maps.append(n3)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4, n4 = self._apply_ddsr(x4, 3, return_noise=True)
        if n4 is not None:
            noise_maps.append(n4)

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
        noise_intensity = self._merge_noise_maps(noise_maps, target_size=d2.shape[-2:])
        if noise_intensity is not None:
            # Keep the noise map as a read-only prior for BUGR.
            noise_prior = noise_intensity.detach()
        else:
            noise_prior = None
        pred_refined, boundary_map, uncertainty_map = self.bugr(
            d2,
            pred_main,
            noise_intensity=noise_prior,
        )

        if return_aux is None:
            return_aux = self.training

        if return_aux:
            return {
                "seg": pred_refined,
                "pred_main": pred_main,
                "pred_refined": pred_refined,
                "boundary_map": boundary_map,
                "uncertainty_map": uncertainty_map,
                "noise_intensity": noise_intensity,
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
