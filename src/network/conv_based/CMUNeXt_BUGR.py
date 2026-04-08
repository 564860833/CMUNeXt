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
    eroded = F.conv2d(flat_mask.float(), kernel, padding=kernel_size // 2) == float(kernel_size * kernel_size)
    boundary = flat_mask & (~eroded)
    has_boundary = boundary.flatten(1).any(dim=1).view(-1, 1, 1, 1)
    boundary = torch.where(has_boundary, boundary, flat_mask).float()
    boundary = boundary.view_as(mask)

    if original_dim == 2:
        return boundary[0, 0]
    if original_dim == 3:
        return boundary.squeeze(1)
    return boundary


# =====================================================================
# 原始 CMUNeXt 基础模块（保持不变）
# =====================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# =====================================================================
# 改进版 BUGR v2
# 改动汇总:
#   [改动1] attention 输入从 (U,B) → (F_compressed, U, B)
#   [改动2] boundary_map 从固定 Sobel → 可学习分支 + Sobel 先验输入
#   [改动3] refinement 从独立预测头 → 残差修正 pred_refined = pred_main + δ
# =====================================================================

class BUGR(nn.Module):
    """
    Boundary Uncertainty-Guided Refinement Module (v2)

    改进设计：
        1. 可学习边界分支：decoder feature + Sobel先验 → 边界图（取代纯 Sobel）
        2. 三源注意力融合：压缩特征 F_d + 不确定性 U + 边界 B → 注意力 A
        3. 残差式精细化：pred_refined = pred_main + δ，δ 初始化为 ~0
    """

    def __init__(self, ch_in, num_classes=1, compress_ch=4):
        """
        Args:
            ch_in:       decoder 最终特征通道数 (dims[0])
            num_classes: 分割类别数
            compress_ch: 特征压缩后的通道数
        """
        super(BUGR, self).__init__()
        self.compress_ch = compress_ch

        # -------- 固定 Sobel 核（仅作为先验输入，不直接做监督输出）--------
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # -------- [改动2] 可学习边界分支 --------
        # 输入：decoder feature (ch_in) + Sobel 先验 (1ch) = ch_in + 1
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(ch_in + 1, ch_in // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 2, ch_in // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # -------- [改动1] 特征压缩器：ch_in → compress_ch --------
        self.feat_compress = nn.Sequential(
            nn.Conv2d(ch_in, compress_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(compress_ch),
            nn.ReLU(inplace=True)
        )

        # -------- [改动1] 三源注意力融合 --------
        # 输入：compressed_feat (compress_ch) + uncertainty (1) + boundary (1)
        attn_in_ch = compress_ch + 1 + 1  # = 6 by default
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(attn_in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # -------- [改动3] 残差精细化头：输出 δ（修正量） --------
        self.refine_head = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // 2, num_classes, kernel_size=1, bias=True)
        )
        # 关键：将最后一层初始化为零
        # 使得训练初期 δ ≈ 0，pred_refined ≈ pred_main，不会破坏 baseline 性能
        nn.init.constant_(self.refine_head[-1].weight, 0)
        nn.init.constant_(self.refine_head[-1].bias, 0)

    def _compute_uncertainty(self, pred):
        """像素级二值信息熵，归一化到 [0,1]"""
        p = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
        return entropy / 0.6931

    def _compute_sobel_prior(self, pred):
        """Sobel 梯度幅值，仅作为边界分支的先验输入"""
        B, C, H, W = pred.shape
        boundary_list = []
        for c in range(C):
            p_c = pred[:, c:c+1, :, :]
            gx = F.conv2d(p_c, self.sobel_x, padding=1)
            gy = F.conv2d(p_c, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            boundary_list.append(grad_mag)
        sobel_out = torch.cat(boundary_list, dim=1).mean(dim=1, keepdim=True)
        # 逐样本归一化
        b_max = sobel_out.flatten(2).max(dim=2)[0].reshape(B, 1, 1, 1) + 1e-8
        return sobel_out / b_max

    def forward(self, f_decoder, pred_main_logits):
        """
        Args:
            f_decoder:        decoder 最终特征 (B, ch_in, H, W)
            pred_main_logits: 主预测 raw logits (B, num_classes, H, W)

        Returns:
            pred_refined:    精细化 logits = pred_main + δ
            boundary_map:    可学习边界图 [0,1]
            uncertainty_map: 像素级信息熵 [0,1]
        """
        pred_sigmoid = torch.sigmoid(pred_main_logits)

        # Step 1: 不确定性图
        uncertainty_map = self._compute_uncertainty(pred_sigmoid)  # (B,1,H,W)

        # Step 2: [改动2] Sobel 先验 → 可学习边界分支
        sobel_prior = self._compute_sobel_prior(pred_sigmoid)     # (B,1,H,W)
        boundary_input = torch.cat([f_decoder, sobel_prior], dim=1)  # (B, ch_in+1, H,W)
        boundary_map = self.boundary_branch(boundary_input)       # (B,1,H,W) 可学习

        # Step 3: [改动1] 三源注意力融合
        f_compressed = self.feat_compress(f_decoder)               # (B, compress_ch, H,W)
        attn_input = torch.cat([f_compressed, uncertainty_map, boundary_map], dim=1)
        attention = self.attention_fusion(attn_input)               # (B,1,H,W)

        # Step 4: 特征增强 → 残差 δ
        f_refined = f_decoder * (1.0 + attention)
        delta = self.refine_head(f_refined)                        # (B, num_classes, H,W)

        # Step 5: [改动3] 残差修正
        pred_refined = pred_main_logits + delta

        return pred_refined, boundary_map, uncertainty_map


# =====================================================================
# [改动1] 分割损失：BCE + Dice
# =====================================================================

class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 联合损失

    - BCE：像素级分类，对每个像素独立惩罚
    - Dice：区域级重叠度，天然抗类别不平衡
    - 两者互补：BCE 保持梯度稳定，Dice 直接优化评价指标
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def _dice_loss(self, logits, target):
        pred = torch.sigmoid(logits)
        pred_flat = pred.flatten(1)       # (B, H*W)
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
        super(ProbBCEDiceLoss, self).__init__()
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


# =====================================================================
# 总损失：BUGRLoss v2
# =====================================================================

class BUGRLoss(nn.Module):
    """
    L_total = L_main(P, GT) + λ₁·L_refine(P', GT) + λ₂·L_boundary(B_learned, GT_edge)

    改进：
        - L_main 和 L_refine 均使用 BCE + Dice
        - L_boundary 监督可学习边界分支（而非 Sobel 输出）
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, edge_kernel_size=3,
                 bce_weight=0.5, dice_weight=0.5,
                 boundary_bce_weight=0.5, boundary_dice_weight=0.5):
        super(BUGRLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.edge_k = edge_kernel_size

        # 主/精细化 分割损失：BCE + Dice
        self.seg_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        # 边界损失：BCE（boundary_map 已 sigmoid，用普通 BCE）
        self.bdy_loss = ProbBCEDiceLoss(
            bce_weight=boundary_bce_weight,
            dice_weight=boundary_dice_weight,
        )

    def _extract_gt_edge(self, gt_mask):
        """形态学 dilation - erosion 提取 GT 边缘"""
        return build_inner_boundary_target(gt_mask, kernel_size=self.edge_k).float()

    def forward(self, pred_main, pred_refined, boundary_map, gt_mask):
        """
        Args:
            pred_main:    主预测 logits      (B,1,H,W)
            pred_refined: 精细化预测 logits   (B,1,H,W) = pred_main + δ
            boundary_map: 可学习边界图 [0,1]  (B,1,H,W)
            gt_mask:      GT 掩码 {0,1}      (B,1,H,W)

        Returns:
            loss_total, loss_dict
        """
        gt = gt_mask.float()

        l_main   = self.seg_loss(pred_main, gt)
        l_refine = self.seg_loss(pred_refined, gt)

        gt_edge    = self._extract_gt_edge(gt)
        l_boundary = self.bdy_loss(boundary_map, gt_edge)

        loss_total = l_main + self.lambda1 * l_refine + self.lambda2 * l_boundary

        loss_dict = {
            'loss_total':    loss_total.item(),
            'loss_main':     l_main.item(),
            'loss_refine':   l_refine.item(),
            'loss_boundary': l_boundary.item(),
        }
        return loss_total, loss_dict


# =====================================================================
# 完整网络：CMUNeXt + BUGR v2
# =====================================================================

class CMUNeXt_BUGR(nn.Module):
    def __init__(self, input_channel=3, num_classes=1,
                 dims=[16, 32, 128, 160, 256],
                 depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_BUGR, self).__init__()

        # ==================== Encoder ====================
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # ==================== Decoder ====================
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        # ==================== 主预测头 ====================
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        # ==================== BUGR v2 ====================
        self.bugr = BUGR(ch_in=dims[0], num_classes=num_classes)

    def forward(self, x, return_aux=None):
        # ---- Encoder ----
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

        # ---- Decoder ----
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)          # (B, dims[0], H, W)

        # ---- 主预测 ----
        pred_main = self.Conv_1x1(d2)   # raw logits

        # ---- BUGR v2 精细化 ----
        # 注意：现在传入 logits（不是 sigmoid），BUGR 内部自行 sigmoid
        pred_refined, boundary_map, uncertainty_map = self.bugr(d2, pred_main)

        if return_aux is None:
            return_aux = self.training

        if return_aux:
            return {
                'seg':             pred_refined,
                'pred_main':       pred_main,
                'pred_refined':    pred_refined,
                'boundary_map':    boundary_map,
                'uncertainty_map': uncertainty_map,
            }
        return pred_refined


# =====================================================================
# 工厂函数
# =====================================================================

def cmunext_bugr(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR(input_channel=input_channel, num_classes=num_classes,
                        dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7])

def cmunext_bugr_s(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR(input_channel=input_channel, num_classes=num_classes,
                        dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9])

def cmunext_bugr_l(input_channel=3, num_classes=1):
    return CMUNeXt_BUGR(input_channel=input_channel, num_classes=num_classes,
                        dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7])


# =====================================================================
# 测试 & 参数统计
# =====================================================================

if __name__ == '__main__':
    model = cmunext_bugr(input_channel=3, num_classes=1).cuda()
    criterion = BUGRLoss(lambda1=1.0, lambda2=0.5, edge_kernel_size=3).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    images = torch.randn(2, 3, 256, 256).cuda()
    masks  = torch.randint(0, 2, (2, 1, 256, 256)).float().cuda()

    # ---- 训练 ----
    model.train()
    outputs = model(images)
    loss, loss_dict = criterion(
        pred_main    = outputs['pred_main'],
        pred_refined = outputs['pred_refined'],
        boundary_map = outputs['boundary_map'],
        gt_mask      = masks,
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Loss: {loss_dict}")

    # ---- 推理 ----
    model.eval()
    with torch.no_grad():
        logits = model(images)
        pred   = torch.sigmoid(logits)
        binary = (pred > 0.5).float()

    # ---- 参数统计 ----
    total_p = sum(p.numel() for p in model.parameters())
    bugr_p  = sum(p.numel() for p in model.bugr.parameters())
    print(f"Total:  {total_p:,}")
    print(f"BUGR:   {bugr_p:,} ({100*bugr_p/total_p:.2f}%)")
    print(f"Output: {logits.shape}")
