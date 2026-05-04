import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic CMUNeXt modules
# ============================================================
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
        groups = 2 if ch_in % 2 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=groups, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1, bias=True),
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
                                bias=True,
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1, bias=True),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1, bias=True),
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


# ============================================================
# AMSE: Adaptive Multi-scale Structure Estimator
# ============================================================
class AdaptiveMultiScaleStructureEstimator(nn.Module):
    """
    Adaptive Multi-scale Structure Estimator (AMSE).

    Scheme-B upgrade:
      - Build several low-frequency structure candidates with AvgPool kernels.
      - Predict a spatial softmax weight for each scale.
      - Use the weighted structure for structure-detail decomposition.

    This replaces the fixed single-scale AvgPool structure estimator.
    """

    def __init__(self, channels, kernels=(3, 5, 7, 9), hidden_ratio=4):
        super().__init__()
        if len(kernels) < 2:
            raise ValueError("AMSE requires at least two kernels.")
        for k in kernels:
            if k % 2 == 0 or k < 3:
                raise ValueError(f"AMSE kernels must be odd and >= 3, got {k}")

        self.kernels = tuple(kernels)
        self.pools = nn.ModuleList(
            [
                nn.AvgPool2d(
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    count_include_pad=False,
                )
                for k in self.kernels
            ]
        )

        hidden = max(channels // hidden_ratio, 4)
        self.scale_selector = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, len(self.kernels), kernel_size=1, bias=True),
        )

    def forward(self, x, return_weights=False):
        structs = [pool(x) for pool in self.pools]
        struct_stack = torch.stack(structs, dim=1)  # B, K, C, H, W

        scale_logits = self.scale_selector(x)       # B, K, H, W
        scale_weights = torch.softmax(scale_logits, dim=1)
        struct = torch.sum(struct_stack * scale_weights.unsqueeze(2), dim=1)

        if return_weights:
            return struct, scale_weights
        return struct


# ============================================================
# AM-BP-DDSR: Adaptive Multi-scale Boundary-Preserved DDSR
# ============================================================
class BPDDSR(nn.Module):
    """
    Adaptive Multi-scale Boundary-Preserved Dual-Domain Speckle Refinement.

    This version includes:
      1) Adaptive multi-scale structure estimation A(x).
      2) Spatially adaptive speckle confidence map S(x).
      3) Boundary preservation gate B(x).

    Final correction:
        y = x - alpha * S(x) * (1 - B(x)) * N(x)

    where:
        N(x): dual-domain predicted speckle component,
        S(x): spatial-channel speckle confidence map,
        B(x): spatial-channel boundary protection map,
        alpha: learnable per-channel positive residual scale.
    """

    def __init__(
        self,
        channels,
        smooth_k=5,
        structure_kernels=(3, 5, 7, 9),
        structure_hidden_ratio=4,
        alpha_init_raw=-5.3,
        hidden_ratio=2,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        hidden = max(channels // hidden_ratio, 8)

        # Scheme B: adaptive multi-scale structure estimator.
        self.structure_estimator = AdaptiveMultiScaleStructureEstimator(
            channels=channels,
            kernels=structure_kernels,
            hidden_ratio=structure_hidden_ratio,
        )

        # Stable local statistics for CV and gradient normalization.
        self.stat_pool = nn.AvgPool2d(
            kernel_size=smooth_k,
            stride=1,
            padding=smooth_k // 2,
            count_include_pad=False,
        )

        # Linear-domain residual branch.
        self.lin_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Log-domain residual branch.
        self.log_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Dual-domain speckle/noise predictor N(x).
        self.noise_pred = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Spatially adaptive speckle confidence S(x).
        self.speckle_map = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Boundary protection gate B(x).
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Per-channel positive residual scale.
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), alpha_init_raw))

        # Fixed Sobel kernels used only for boundary evidence.
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @staticmethod
    def signed_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _local_cv(self, x):
        mean = self.stat_pool(x)
        mean_sq = self.stat_pool(x * x)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        mean_abs = self.stat_pool(torch.abs(x))
        return std / (mean_abs + self.eps)

    def _gradient_magnitude(self, x):
        c = x.shape[1]
        weight_x = self.sobel_x.expand(c, 1, 3, 3)
        weight_y = self.sobel_y.expand(c, 1, 3, 3)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=c)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=c)
        grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        grad_norm = grad / (self.stat_pool(grad) + self.eps)
        return grad_norm

    def forward(self, x, return_maps=False):
        # Linear-domain adaptive multi-scale structure-detail decomposition.
        struct, struct_weights = self.structure_estimator(x, return_weights=True)
        detail = x - struct
        lin_feat = self.lin_branch(detail)

        # Log-domain adaptive multi-scale structure-detail decomposition.
        x_log = self.signed_log(x)
        struct_log, struct_log_weights = self.structure_estimator(x_log, return_weights=True)
        log_detail = x_log - struct_log
        log_feat = self.log_branch(log_detail)

        # Predict dual-domain speckle component N(x).
        noise = self.noise_pred(torch.cat([lin_feat, log_feat], dim=1))

        # Spatially adaptive speckle confidence S(x).
        cv = self._local_cv(x)
        speckle = self.speckle_map(torch.cat([detail, log_detail, cv], dim=1))

        # Boundary preservation gate B(x).
        grad = self._gradient_magnitude(x)
        boundary = self.boundary_gate(
            torch.cat([torch.abs(detail), torch.abs(log_detail), grad], dim=1)
        )

        # Boundary-preserved adaptive residual correction.
        alpha = F.softplus(self.alpha)
        correction = alpha * speckle * (1.0 - boundary) * noise
        out = x - correction

        if return_maps:
            return out, {
                "noise": noise,
                "speckle_map": speckle,
                "boundary_gate": boundary,
                "alpha": alpha,
                "correction": correction,
                "structure_weights": struct_weights,
                "log_structure_weights": struct_log_weights,
            }
        return out


# ============================================================
# CMUNeXt with AM-BP-DDSR
# ============================================================
class CMUNeXt_BPDDSR(nn.Module):
    """
    CMUNeXt with Adaptive Multi-scale Boundary-Preserved Dual-Domain
    Speckle Refinement.

    Default stages are shallow stages (0, 1), which correspond to high-resolution
    low-level ultrasound features where speckle and boundary details are prominent.
    """

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        bpddsr_stages=(0, 1),
        bpddsr_smooth_k=5,
        structure_kernels=(3, 5, 7, 9),
        structure_hidden_ratio=4,
        alpha_init_raw=-5.3,
        hidden_ratio=2,
    ):
        super().__init__()
        self.bpddsr_stages = set(bpddsr_stages)

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # AM-BP-DDSR modules for selected skip stages.
        skip_dims = [dims[0], dims[1], dims[2], dims[3]]
        self.bpddsr_modules = nn.ModuleDict()
        for s in bpddsr_stages:
            if s < 0 or s >= len(skip_dims):
                raise ValueError(f"bpddsr stage must be in [0, 3], got {s}")
            self.bpddsr_modules[str(s)] = BPDDSR(
                channels=skip_dims[s],
                smooth_k=bpddsr_smooth_k,
                structure_kernels=structure_kernels,
                structure_hidden_ratio=structure_hidden_ratio,
                alpha_init_raw=alpha_init_raw,
                hidden_ratio=hidden_ratio,
            )

        # Decoder unchanged from CMUNeXt.
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def _apply_bpddsr(self, x, stage):
        if stage in self.bpddsr_stages:
            return self.bpddsr_modules[str(stage)](x)
        return x

    def forward(self, x):
        # Encode + AM-BP-DDSR
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x1 = self._apply_bpddsr(x1, 0)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x2 = self._apply_bpddsr(x2, 1)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3 = self._apply_bpddsr(x3, 2)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4 = self._apply_bpddsr(x4, 3)

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # Decode, unchanged from CMUNeXt.
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        return self.Conv_1x1(d2)


# ============================================================
# Factory functions
# ============================================================
def cmunext_bpddsr(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    bpddsr_stages=(0, 1),
    bpddsr_smooth_k=5,
    structure_kernels=(3, 5, 7, 9),
    structure_hidden_ratio=4,
    alpha_init_raw=-5.3,
    hidden_ratio=2,
):
    return CMUNeXt_BPDDSR(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        bpddsr_stages=bpddsr_stages,
        bpddsr_smooth_k=bpddsr_smooth_k,
        structure_kernels=structure_kernels,
        structure_hidden_ratio=structure_hidden_ratio,
        alpha_init_raw=alpha_init_raw,
        hidden_ratio=hidden_ratio,
    )


def cmunext_bpddsr_shallow(input_channel=3, num_classes=1):
    return cmunext_bpddsr(
        input_channel=input_channel,
        num_classes=num_classes,
        bpddsr_stages=(0, 1),
    )


def cmunext_bpddsr_deep(input_channel=3, num_classes=1):
    return cmunext_bpddsr(
        input_channel=input_channel,
        num_classes=num_classes,
        bpddsr_stages=(2, 3),
    )


def cmunext_bpddsr_s(input_channel=3, num_classes=1):
    return cmunext_bpddsr(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        bpddsr_stages=(0, 1),
    )


def cmunext_bpddsr_l(input_channel=3, num_classes=1):
    return cmunext_bpddsr(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        bpddsr_stages=(0, 1),
    )


# Backward-friendly aliases.
cmunext_bpd = cmunext_bpddsr
cmunext_bp_ddsr = cmunext_bpddsr
cmunext_ambpddsr = cmunext_bpddsr


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    model = cmunext_bpddsr(input_channel=3, num_classes=1).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        y = model(x)

    total_p = sum(p.numel() for p in model.parameters())
    bpddsr_p = sum(p.numel() for p in model.bpddsr_modules.parameters())

    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    print(f"Total params : {total_p:,}")
    print(f"AM-BP-DDSR params: {bpddsr_p:,} ({100 * bpddsr_p / total_p:.2f}%)")

    # Optional map sanity check on one AM-BP-DDSR module.
    if "0" in model.bpddsr_modules:
        feat = model.encoder1(model.stem(x))
        _, maps = model.bpddsr_modules["0"](feat, return_maps=True)
        print("Speckle map shape     :", tuple(maps["speckle_map"].shape))
        print("Boundary gate shape   :", tuple(maps["boundary_gate"].shape))
        print("Structure weights shape:", tuple(maps["structure_weights"].shape))
