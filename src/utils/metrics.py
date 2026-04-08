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


def _prepare_prediction(output, from_logits=True):
    if torch.is_tensor(output):
        output = output.detach().cpu().float()
        if from_logits:
            output = torch.sigmoid(output)
    else:
        output = torch.as_tensor(output, dtype=torch.float32)
    return output


def _prepare_target(target):
    if torch.is_tensor(target):
        return target.detach().cpu().float()
    return torch.as_tensor(target, dtype=torch.float32)


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


def iou_score(output, target, threshold=0.5, from_logits=True):
    smooth = 1e-5

    output = _prepare_prediction(output, from_logits=from_logits)
    target = _prepare_target(target)
    output_ = output > threshold
    target_ = target > 0.5

    intersection = (output_ & target_).sum().item()
    union = (output_ | target_).sum().item()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)

    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return iou, dice, SE, PC, F1, SP, ACC


def boundary_scores(output, target, threshold=0.5, from_logits=True):
    output = _prepare_prediction(output, from_logits=from_logits) > threshold
    target = _prepare_target(target) > 0.5

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


def _iter_batches(outputs, targets):
    if isinstance(outputs, (list, tuple)):
        if not isinstance(targets, (list, tuple)) or len(outputs) != len(targets):
            raise ValueError("outputs and targets must be lists of the same length.")
        for output_batch, target_batch in zip(outputs, targets):
            yield output_batch, target_batch
        return

    yield outputs, targets


def _batch_size(batch):
    if torch.is_tensor(batch):
        if batch.dim() == 0:
            return 1
        return int(batch.shape[0])
    return int(np.asarray(batch).shape[0])


def find_best_threshold(outputs, targets, thresholds, select_metric="iou", from_logits=True):
    if select_metric not in {"iou", "f1"}:
        raise ValueError("select_metric must be 'iou' or 'f1'.")

    thresholds = [float(threshold) for threshold in thresholds]
    if not thresholds:
        raise ValueError("thresholds must not be empty.")

    best_metrics = None
    metric_keys = ("iou", "dice", "se", "pc", "f1", "sp", "acc")

    for threshold in thresholds:
        totals = {key: 0.0 for key in metric_keys}
        total_count = 0

        for output_batch, target_batch in _iter_batches(outputs, targets):
            iou, dice, se, pc, f1, sp, acc = iou_score(
                output_batch,
                target_batch,
                threshold=threshold,
                from_logits=from_logits,
            )
            batch_count = _batch_size(target_batch)
            total_count += batch_count
            totals["iou"] += iou * batch_count
            totals["dice"] += dice * batch_count
            totals["se"] += se * batch_count
            totals["pc"] += pc * batch_count
            totals["f1"] += f1 * batch_count
            totals["sp"] += sp * batch_count
            totals["acc"] += acc * batch_count

        current = {key: totals[key] / max(total_count, 1) for key in metric_keys}
        current["threshold"] = threshold

        if best_metrics is None:
            best_metrics = current
            continue

        best_score = best_metrics[select_metric]
        current_score = current[select_metric]
        if current_score > best_score + 1e-8:
            best_metrics = current
            continue
        if abs(current_score - best_score) <= 1e-8 and abs(threshold - 0.5) < abs(best_metrics["threshold"] - 0.5):
            best_metrics = current

    hd95_total = 0.0
    assd_total = 0.0
    total_count = 0
    for output_batch, target_batch in _iter_batches(outputs, targets):
        hd95, assd = boundary_scores(
            output_batch,
            target_batch,
            threshold=best_metrics["threshold"],
            from_logits=from_logits,
        )
        batch_count = _batch_size(target_batch)
        total_count += batch_count
        hd95_total += hd95 * batch_count
        assd_total += assd * batch_count

    best_metrics["hd95"] = hd95_total / max(total_count, 1)
    best_metrics["assd"] = assd_total / max(total_count, 1)
    return best_metrics


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
