import numpy as np
import torch
import torch.nn.functional as F


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    
    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return iou, dice, SE, PC, F1, SP, ACC


def _mask_to_boundary(mask):
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    mask = mask.bool()
    if not mask.any():
        return mask

    mask_float = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), dtype=mask_float.dtype, device=mask_float.device)
    eroded = F.conv2d(mask_float, kernel, padding=1) == 9
    eroded = eroded.squeeze(0).squeeze(0)
    boundary = mask & (~eroded)
    return boundary if boundary.any() else mask


def boundary_scores(output, target):
    if torch.is_tensor(output):
        output = (torch.sigmoid(output).detach().cpu() > 0.5)
    else:
        output = torch.tensor(output > 0.5)

    if torch.is_tensor(target):
        target = (target.detach().cpu() > 0.5)
    else:
        target = torch.tensor(target > 0.5)

    hd95_list = []
    assd_list = []

    for pred_mask, gt_mask in zip(output, target):
        pred_mask = pred_mask.squeeze(0).bool()
        gt_mask = gt_mask.squeeze(0).bool()

        pred_boundary = _mask_to_boundary(pred_mask)
        gt_boundary = _mask_to_boundary(gt_mask)
        pred_points = torch.nonzero(pred_boundary, as_tuple=False).float()
        gt_points = torch.nonzero(gt_boundary, as_tuple=False).float()

        h, w = pred_mask.shape[-2:]
        fallback = float((h ** 2 + w ** 2) ** 0.5)

        if pred_points.numel() == 0 and gt_points.numel() == 0:
            distances = np.array([0.0], dtype=np.float32)
        elif pred_points.numel() == 0 or gt_points.numel() == 0:
            distances = np.array([fallback], dtype=np.float32)
        else:
            pairwise = torch.cdist(pred_points, gt_points, p=2)
            pred_to_gt = pairwise.min(dim=1).values.cpu().numpy()
            gt_to_pred = pairwise.min(dim=0).values.cpu().numpy()
            distances = np.concatenate([pred_to_gt, gt_to_pred], axis=0)

        hd95_list.append(float(np.percentile(distances, 95)))
        assd_list.append(float(distances.mean()))

    return float(np.mean(hd95_list)), float(np.mean(assd_list))


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
