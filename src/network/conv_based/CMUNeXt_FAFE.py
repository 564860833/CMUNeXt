import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Base CMUNeXt modules
# =====================================================================

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


# =====================================================================
# FAFE final version
# =====================================================================

class ChannelAttention(nn.Module):
    """SE-style channel attention for the high-frequency branch."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class LearnableFrequencyMask(nn.Module):
    """
    Learnable radial low-pass mask after fftshift.

    mask(r) = sigmoid(sharpness * (radius - r))
    r < radius  -> low frequency passes
    r > radius  -> high frequency passes through 1 - mask
    """

    def __init__(self, init_radius_ratio=0.25, init_sharpness=10.0):
        super().__init__()
        self.radius_ratio = nn.Parameter(torch.tensor(float(init_radius_ratio)))
        self.sharpness = nn.Parameter(torch.tensor(float(init_sharpness)))

    def forward(self, H, W, device):
        cy, cx = H / 2.0, W / 2.0
        y = torch.arange(H, device=device, dtype=torch.float32) - cy
        x = torch.arange(W, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        max_dist = math.sqrt(cy ** 2 + cx ** 2) + 1e-6
        dist = torch.sqrt(yy ** 2 + xx ** 2) / max_dist

        radius = torch.clamp(self.radius_ratio, 0.05, 0.95)
        sharpness = torch.clamp(self.sharpness, 1.0, 50.0)
        mask = torch.sigmoid(sharpness * (radius - dist))
        return mask.unsqueeze(0).unsqueeze(0)


class FAFE(nn.Module):
    """
    Frequency-Aware Feature Enhancement.

    Final design choices:
      1) true safe residual: out = x + tanh(gate) * delta
      2) no early ReLU inside low/high branches to preserve signed responses
      3) frequency visualization info is collected only on demand
    """

    def __init__(self, channels, ca_reduction=4, init_radius=0.25, init_sharpness=10.0):
        super().__init__()
        self.freq_mask = LearnableFrequencyMask(
            init_radius_ratio=init_radius,
            init_sharpness=init_sharpness,
        )

        # Branch refiners keep signed information.
        self.low_freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.high_freq_attn = ChannelAttention(channels, reduction=ca_reduction)
        self.high_freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Merge low/high residual evidence into one signed delta.
        self.delta_merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.delta_post = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Zero-init gate -> exact identity at initialization.
        self.gate = nn.Parameter(torch.zeros(1))

        self._collect_info = False
        self.last_freq_info = None

    def set_collect_info(self, flag: bool):
        self._collect_info = bool(flag)
        if not self._collect_info:
            self.last_freq_info = None

    def forward(self, x):
        identity = x
        _, _, H, W = x.shape

        # Spatial -> frequency
        x_freq = torch.fft.fft2(x, norm="ortho")
        x_freq_shifted = torch.fft.fftshift(x_freq, dim=(-2, -1))

        # Low/high masks
        mask_low = self.freq_mask(H, W, x.device)
        mask_high = 1.0 - mask_low

        # Frequency split
        x_low_freq = x_freq_shifted * mask_low
        x_high_freq = x_freq_shifted * mask_high

        # Frequency -> spatial
        x_low = torch.fft.ifft2(
            torch.fft.ifftshift(x_low_freq, dim=(-2, -1)), norm="ortho"
        ).real
        x_high = torch.fft.ifft2(
            torch.fft.ifftshift(x_high_freq, dim=(-2, -1)), norm="ortho"
        ).real

        # Signed branch processing
        x_low = self.low_freq_refine(x_low)
        x_high = self.high_freq_attn(x_high)
        x_high = self.high_freq_refine(x_high)

        # Signed residual increment
        delta = self.delta_merge(torch.cat([x_low, x_high], dim=1))
        delta = self.delta_post(delta)

        gate = torch.tanh(self.gate)
        x_out = identity + gate * delta

        if self._collect_info:
            self.last_freq_info = {
                "mask_low": mask_low.detach().cpu(),
                "mask_high": mask_high.detach().cpu(),
                "radius": float(self.freq_mask.radius_ratio.detach().cpu()),
                "sharpness": float(self.freq_mask.sharpness.detach().cpu()),
                "gate": float(gate.detach().cpu()),
                "delta_norm": float(delta.detach().norm().cpu()),
                "spectrum_magnitude": x_freq_shifted.abs().mean(dim=1, keepdim=True).detach().cpu(),
            }
        else:
            self.last_freq_info = None

        return x_out


class FAFEWrapper(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.fafe = FAFE(channels, **kwargs)

    def forward(self, x):
        return self.fafe(x)

    def set_collect_info(self, flag: bool):
        self.fafe.set_collect_info(flag)


# =====================================================================
# Loss
# =====================================================================

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


# =====================================================================
# CMUNeXt + FAFE
# =====================================================================

class CMUNeXt_FAFE(nn.Module):
    """
    FAFE insertion strategies:
      - bottleneck : encoder5 only (recommended default)
      - encoder    : all encoder stages
      - decoder    : all decoder stages
      - all        : encoder + decoder all stages
      - last3_enc  : encoder3, encoder4, encoder5
      - enc45_dec  : encoder4, encoder5 and all decoder stages
    """

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        fafe_positions="bottleneck",
        fafe_init_radius=0.25,
        fafe_init_sharpness=10.0,
    ):
        super().__init__()
        self.fafe_positions = fafe_positions

        fafe_kwargs = dict(
            init_radius=fafe_init_radius,
            init_sharpness=fafe_init_sharpness,
        )

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.fafe_modules = nn.ModuleDict()
        self._build_fafe_modules(dims, fafe_positions, fafe_kwargs)

    def _build_fafe_modules(self, dims, positions, kwargs):
        encoder_points = {
            "enc1": dims[0],
            "enc2": dims[1],
            "enc3": dims[2],
            "enc4": dims[3],
            "enc5": dims[4],
        }
        decoder_points = {
            "dec5": dims[3],
            "dec4": dims[2],
            "dec3": dims[1],
            "dec2": dims[0],
        }

        if positions == "bottleneck":
            active = {"enc5": encoder_points["enc5"]}
        elif positions == "encoder":
            active = encoder_points
        elif positions == "decoder":
            active = decoder_points
        elif positions == "last3_enc":
            active = {k: encoder_points[k] for k in ["enc3", "enc4", "enc5"]}
        elif positions == "all":
            active = {**encoder_points, **decoder_points}
        elif positions == "enc45_dec":
            active = {
                "enc4": encoder_points["enc4"],
                "enc5": encoder_points["enc5"],
                **decoder_points,
            }
        else:
            raise ValueError(f"Unknown fafe_positions: {positions}")

        for name, ch in active.items():
            self.fafe_modules[name] = FAFEWrapper(ch, **kwargs)

    def _apply_fafe(self, x, name):
        if name in self.fafe_modules:
            return self.fafe_modules[name](x)
        return x

    def set_collect_info(self, flag: bool):
        for module in self.fafe_modules.values():
            module.set_collect_info(flag)

    def get_freq_info(self):
        info = {}
        for name, module in self.fafe_modules.items():
            if module.fafe.last_freq_info is not None:
                info[name] = module.fafe.last_freq_info
        return info

    def get_gate_values(self):
        return {name: float(torch.tanh(module.fafe.gate).detach().cpu()) for name, module in self.fafe_modules.items()}

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x1 = self._apply_fafe(x1, "enc1")

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x2 = self._apply_fafe(x2, "enc2")

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x3 = self._apply_fafe(x3, "enc3")

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x4 = self._apply_fafe(x4, "enc4")

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)
        x5 = self._apply_fafe(x5, "enc5")

        # Decoder
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))
        d5 = self._apply_fafe(d5, "dec5")

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d4 = self._apply_fafe(d4, "dec4")

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d3 = self._apply_fafe(d3, "dec3")

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        d2 = self._apply_fafe(d2, "dec2")

        return self.Conv_1x1(d2)


# =====================================================================
# Factory functions
# =====================================================================

def cmunext_fafe(input_channel=3, num_classes=1, fafe_positions="bottleneck"):
    return CMUNeXt_FAFE(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        fafe_positions=fafe_positions,
    )


def cmunext_fafe_s(input_channel=3, num_classes=1, fafe_positions="bottleneck"):
    return CMUNeXt_FAFE(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        fafe_positions=fafe_positions,
    )


def cmunext_fafe_l(input_channel=3, num_classes=1, fafe_positions="bottleneck"):
    return CMUNeXt_FAFE(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        fafe_positions=fafe_positions,
    )


# =====================================================================
# Lightweight utilities
# =====================================================================

def collect_and_print_freq_info(model, images):
    model.set_collect_info(True)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(images)
    if was_training:
        model.train()
    model.set_collect_info(False)

    info = model.get_freq_info()
    print("\n[FAFE Frequency Analysis]")
    print(f"{'Module':<10} {'Radius':>8} {'Sharp':>8} {'Gate':>8} {'dNorm':>10}")
    print("-" * 48)
    for name, fi in info.items():
        print(
            f"{name:<10} {fi['radius']:>8.4f} {fi['sharpness']:>8.2f} "
            f"{fi['gate']:>8.5f} {fi['delta_norm']:>10.4f}"
        )
    return info


def log_gate_values(model, epoch, writer=None):
    gates = model.get_gate_values()
    print(f"[Epoch {epoch}] Gate values: ", end="")
    for name, val in gates.items():
        print(f"{name}={val:.5f}  ", end="")
        if writer is not None:
            writer.add_scalar(f"FAFE_gate/{name}", val, epoch)
    print()
    return gates


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cmunext_fafe(fafe_positions="bottleneck").to(device)
    criterion = BCEDiceLoss().to(device)

    images = torch.randn(2, 3, 256, 256, device=device)
    masks = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()

    model.train()
    pred = model(images)
    loss = criterion(pred, masks)
    loss.backward()

    print("Output shape:", tuple(pred.shape))
    print("Loss:", float(loss.detach().cpu()))
    print("Gate values:", model.get_gate_values())
