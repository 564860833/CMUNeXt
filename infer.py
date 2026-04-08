import argparse
import os

import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import src.utils.losses as losses
from src.dataloader.dataset import MedicalDataSets
from src.utils.metrics import iou_score, boundary_scores, find_best_threshold
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_BoundaryDS import cmunext_boundaryds
from src.network.conv_based.CMUNeXt_BUGR import cmunext_bugr
from src.network.conv_based.CMUNeXt_BUGR_SpeckleEnhance import cmunext_bugr_speckleenhance
from src.network.conv_based.CMUNeXt_DistanceAux import cmunext_distanceaux
from src.network.conv_based.CMUNeXt_DualGAG import cmunext_dualgag
from src.network.conv_based.CMUNeXt_DualGAG_DistanceAux import cmunext_dualgag_distanceaux
from src.network.conv_based.CMUNeXt_PresenceAux import cmunext_presenceaux
from src.network.conv_based.CMUNeXt_SpeckleEnhance import cmunext_speckle
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNeXt import UNext
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


def build_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_PresenceAux":
        model = cmunext_presenceaux(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_BUGR":
        model = cmunext_bugr(num_classes=args.num_classes)
    elif args.model == "BUGR_SpeckleEnhance":
        model = cmunext_bugr_speckleenhance(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_BoundaryDS":
        model = cmunext_boundaryds(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_DistanceAux":
        model = cmunext_distanceaux(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_DualGAG":
        model = cmunext_dualgag(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_DualGAG_DistanceAux":
        model = cmunext_dualgag_distanceaux(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_SpeckleEnhance":
        model = cmunext_speckle(num_classes=args.num_classes)
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes)
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes)
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes)
    else:
        print(f"Attempting to load transformer-based model: {args.model}")
        model = get_transformer_based_model(
            model_name=args.model,
            img_size=args.img_size,
            num_classes=args.num_classes,
            in_ch=3,
        )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    return model


def load_model(model_path, args, device):
    model = build_model(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def forward_with_model(model, model_name, x):
    if model_name in {
        "CMUNeXt_PresenceAux",
        "CMUNeXt_BUGR",
        "BUGR_SpeckleEnhance",
        "CMUNeXt_BoundaryDS",
        "CMUNeXt_DistanceAux",
        "CMUNeXt_DualGAG_DistanceAux",
    }:
        return model(x, return_aux=False)
    return model(x)


def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])


def build_validation_thresholds(args):
    if args.val_threshold_mode == "fixed":
        thresholds = [args.val_threshold]
    else:
        if args.val_threshold_step <= 0:
            raise ValueError("val_threshold_step must be positive.")
        if args.val_threshold_start > args.val_threshold_end:
            raise ValueError("val_threshold_start must be <= val_threshold_end.")
        thresholds = np.arange(
            args.val_threshold_start,
            args.val_threshold_end + args.val_threshold_step * 0.5,
            args.val_threshold_step,
        )

    thresholds = [
        round(float(np.clip(threshold, 1e-4, 1.0 - 1e-4)), 4)
        for threshold in thresholds
    ]
    return sorted(set(thresholds))


def validate(model, val_loader, criterion, device, args, save_dir="validation_results"):
    model.eval()
    val_loss = 0.0
    prob_batches = []
    target_batches = []
    metrics = None
    threshold = args.val_threshold
    totals = {
        "iou": 0.0,
        "dice": 0.0,
        "se": 0.0,
        "pc": 0.0,
        "f1": 0.0,
        "sp": 0.0,
        "hd95": 0.0,
        "assd": 0.0,
        "acc": 0.0,
    }
    total_count = 0
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = forward_with_model(model, args.model, img_batch)
            loss = criterion(outputs, label_batch)

            val_loss += loss.item()
            if args.val_threshold_mode == "scan":
                prob_batches.append(torch.sigmoid(outputs).detach().cpu())
                target_batches.append(label_batch.detach().cpu())
                continue

            batch_count = int(img_batch.shape[0])
            iou, dice, se, pc, f1, sp, acc = iou_score(outputs, label_batch, threshold=threshold)
            hd95, assd = boundary_scores(outputs, label_batch, threshold=threshold)
            totals["iou"] += iou * batch_count
            totals["dice"] += dice * batch_count
            totals["se"] += se * batch_count
            totals["pc"] += pc * batch_count
            totals["f1"] += f1 * batch_count
            totals["sp"] += sp * batch_count
            totals["hd95"] += hd95 * batch_count
            totals["assd"] += assd * batch_count
            totals["acc"] += acc * batch_count
            total_count += batch_count

            predicted_masks = (torch.sigmoid(outputs) > threshold).float()
            for idx, img in enumerate(predicted_masks.cpu()):
                save_path = os.path.join(save_dir, f"batch_{i_batch}_img_{idx}.png")
                save_image(img, save_path)

    val_loss /= len(val_loader)
    if args.val_threshold_mode == "scan":
        thresholds = build_validation_thresholds(args)
        metrics = find_best_threshold(
            prob_batches,
            target_batches,
            thresholds,
            select_metric=args.val_threshold_metric,
            from_logits=False,
        )
        threshold = metrics["threshold"]

        for i_batch, probs_batch in enumerate(prob_batches):
            predicted_masks = (probs_batch > threshold).float()
            for idx, img in enumerate(predicted_masks):
                save_path = os.path.join(save_dir, f"batch_{i_batch}_img_{idx}.png")
                save_image(img, save_path)
    else:
        metrics = {
            "threshold": threshold,
            "iou": totals["iou"] / max(total_count, 1),
            "dice": totals["dice"] / max(total_count, 1),
            "se": totals["se"] / max(total_count, 1),
            "pc": totals["pc"] / max(total_count, 1),
            "f1": totals["f1"] / max(total_count, 1),
            "sp": totals["sp"] / max(total_count, 1),
            "hd95": totals["hd95"] / max(total_count, 1),
            "assd": totals["assd"] / max(total_count, 1),
            "acc": totals["acc"] / max(total_count, 1),
        }

    print(
        f"Validation loss: {val_loss:.4f}, threshold: {metrics['threshold']:.4f}, "
        f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, F1: {metrics['f1']:.4f}, "
        f"SE: {metrics['se']:.4f}, PC: {metrics['pc']:.4f}, SP: {metrics['sp']:.4f}, "
        f"HD95: {metrics['hd95']:.4f}, ASSD: {metrics['assd']:.4f}, ACC: {metrics['acc']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")

    model_choices = [
        "CMUNet", "CMUNeXt", "CMUNeXt_PresenceAux", "CMUNeXt_BUGR",
        "BUGR_SpeckleEnhance",
        "CMUNeXt_BoundaryDS", "CMUNeXt_DistanceAux", "CMUNeXt_DualGAG",
        "CMUNeXt_DualGAG_DistanceAux", "CMUNeXt_SpeckleEnhance",
        "U_Net", "AttU_Net", "UNext", "UNetplus", "UNet3plus",
        "TransUnet", "SwinUnet", "MedT", "Mobile_U_ViT",
    ]
    parser.add_argument("--model", type=str, default="U_Net", choices=model_choices, help="model type")
    parser.add_argument("--model_path", type=str, default="./checkpoint/U_Net_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--base_dir", type=str, default="./data/test", help="base directory of dataset")
    parser.add_argument("--train_file_dir", type=str, default="train.txt",
                        help="(Required by MedicalDataSets) train file directory")
    parser.add_argument("--val_file_dir", type=str, default="test_val.txt", help="validation file list")
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--val_threshold_mode", type=str, default="fixed", choices=["fixed", "scan"],
                        help="Use a fixed validation threshold or scan a threshold range")
    parser.add_argument("--val_threshold", type=float, default=0.5,
                        help="Validation threshold when val_threshold_mode=fixed")
    parser.add_argument("--val_threshold_start", type=float, default=0.30,
                        help="Threshold scan start when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_end", type=float, default=0.70,
                        help="Threshold scan end when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_step", type=float, default=0.02,
                        help="Threshold scan step when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_metric", type=str, default="iou", choices=["iou", "f1"],
                        help="Metric used to pick the best validation threshold")
    args = parser.parse_args()

    if args.model == "SwinUnet" and args.img_size == 256:
        print("SwinUnet requires 224x224 input, adjusting img_size.")
        args.img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args, device)

    val_transform = get_val_transform(args.img_size)
    db_val = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=val_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
    )
    val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = losses.__dict__["BCEDiceLoss"]().to(device)
    save_directory = os.path.join(os.path.dirname(args.model_path), f"predictions_{args.model}")
    print(f"Saving predictions to: {save_directory}")
    validate(model, val_loader, criterion, device, args=args, save_dir=save_directory)
