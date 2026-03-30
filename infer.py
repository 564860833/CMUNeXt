import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2 # 注意：原文件中这一行被注释了，保持不变
import src.utils.losses as losses
from src.utils.metrics import iou_score
from torchvision.utils import save_image

# 假设模型导入基于您的训练脚本
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_MKDC import cmunext_mkdc
from src.network.conv_based.CMUNeXt_GAG import cmunext_gag
from src.network.conv_based.CMUNeXt_CMFA import cmunext_cmfa
from src.network.conv_based.CMUNeXt_PresenceAux import cmunext_presenceaux
from src.network.conv_based.CMUNeXt_BoundaryDS import cmunext_boundaryds
from src.network.conv_based.CMUNeXt_DualGAG import cmunext_dualgag
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


def load_model(model_path, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Model selection based on argument
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "CMUNeXt_MKDC":
        model = cmunext_mkdc(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "CMUNeXt_GAG":
        model = cmunext_gag(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "CMUNeXt_CMFA":
        model = cmunext_cmfa(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "CMUNeXt_PresenceAux":
        model = cmunext_presenceaux(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "CMUNeXt_BoundaryDS":
        model = cmunext_boundaryds(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "CMUNeXt_DualGAG":
        model = cmunext_dualgag(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    else:
        # Adjust accordingly for transformer-based models
        # (假设 get_transformer_based_model 知道如何处理 "TransUnet", "SwinUnet" 等)
        print(f"Attempting to load transformer-based model: {args.model}")
        model = get_transformer_based_model(model_name=args.model, img_size=args.img_size, num_classes=args.num_classes,
                                            in_ch=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def forward_with_model(model, model_name, x):
    if model_name in {"CMUNeXt_PresenceAux", "CMUNeXt_BoundaryDS"}:
        return model(x, return_aux=False)
    return model(x)


def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
        # ToTensorV2(), # 保持原样 (被注释)
    ])


def validate(model, val_loader, criterion, device, save_dir="validation_results"):
    """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_rvd = 0.0
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = forward_with_model(model, args.model, img_batch)
            loss = criterion(outputs, label_batch)

            val_loss += loss.item()

            # 使用 metrics.py 的签名
            iou, dice, SE, PC, F1, SP, ACC = iou_score(outputs, label_batch)
            rvd = SE  # 假设 'rvd' 在这里是指 'SE' (Sensitivity/Recall)

            val_iou += iou
            val_dice += dice
            if rvd < 1:  # 保持原 rvd 逻辑
                val_rvd += rvd

            if i_batch % 1 == 0:
                outputs_sig = torch.sigmoid(outputs)
                # 应用阈值
                predicted_masks = (outputs_sig > 0.5).float()

                # 保存图像
                for idx, img in enumerate(predicted_masks.cpu()):
                    # 使用原逻辑命名
                    base_name = f"batch_{i_batch}_img_{idx}.png"
                    save_path = os.path.join(save_dir, base_name)
                    save_image(img, save_path)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    val_rvd /= len(val_loader)
    print(
        f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证dice：{val_dice:.4f}, 验证rvd(SE)：{val_rvd:.4f}')  # 澄清rvd为SE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")

    # 我们将 main.py 中的 transformer 模型也加入列表
    model_choices = [
        "CMUNet", "CMUNeXt", "CMUNeXt_MKDC", "CMUNeXt_GAG", "CMUNeXt_CMFA",
        "CMUNeXt_PresenceAux", "CMUNeXt_BoundaryDS", "CMUNeXt_DualGAG",
        "U_Net", "AttU_Net", "UNext", "UNetplus", "UNet3plus",
        "TransUnet", "SwinUnet", "MedT", "Mobile_U_ViT"
    ]
    parser.add_argument('--model', type=str, default="U_Net",
                        choices=model_choices,
                        help='model type')

    parser.add_argument('--model_path', type=str, default="./checkpoint/U_Net_model.pth",
                        help='Path to the trained model')
    parser.add_argument('--base_dir', type=str, default="./data/test", help='base directory of dataset')

    parser.add_argument('--train_file_dir', type=str, default="train.txt",
                        help='(Required by MedicalDataSets) train file directory')
    parser.add_argument('--val_file_dir', type=str, default="test_val.txt", help='validation file list')

    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    # 自动调整SwinUnet的img_size (从main.py借鉴)
    if args.model == "SwinUnet" and args.img_size == 256:
        print("SwinUnet requires 224x224 input, adjusting img_size.")
        args.img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args, device)

    val_transform = get_val_transform(args.img_size)

    # 确保 MedicalDataSets 获得所需的所有参数
    db_val = MedicalDataSets(base_dir=args.base_dir,
                             split="val",
                             transform=val_transform,
                             train_file_dir=args.train_file_dir,  # 传递默认值
                             val_file_dir=args.val_file_dir)

    val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = losses.__dict__['BCEDiceLoss']().to(device)

    # 定义保存预测图像的目录
    save_directory = os.path.join(os.path.dirname(args.model_path), f"predictions_{args.model}")
    print(f"Saving predictions to: {save_directory}")

    validate(model, val_loader, criterion, device, save_dir=save_directory)
