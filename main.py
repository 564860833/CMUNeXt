import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import logging  # <=== 新增 1: 导入 logging 模块
import sys  # <=== 新增 2: 导入 sys 模块
import matplotlib.pyplot as plt  # <=== 新增 3: 导入 matplotlib

from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, ElasticTransform, GridDistortion, RandomBrightnessContrast, \
    GaussNoise, OneOf

import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score, boundary_scores

from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
# 找到原有导入 CMUNeXt 系列模型的位置，并在其后添加：
from src.network.conv_based.CMUNeXt_MKDC import cmunext_mkdc
from src.network.conv_based.CMUNeXt_GAG import cmunext_gag
from src.network.conv_based.CMUNeXt_CMFA import cmunext_cmfa
from src.network.conv_based.CMUNeXt_PresenceAux import cmunext_presenceaux, PresenceAuxLoss
from src.network.conv_based.CMUNeXt_BoundaryDS import cmunext_boundaryds, BoundaryDeepSupervisionLoss
from src.network.conv_based.CMUNeXt_DistanceAux import cmunext_distanceaux, DistanceAuxLoss
from src.network.conv_based.CMUNeXt_DualGAG import cmunext_dualgag
from src.network.conv_based.CMUNeXt_DualGAG_DistanceAux import cmunext_dualgag_distanceaux
from src.network.conv_based.CMUNeXt_SpeckleEnhance import cmunext_speckle



from src.network.transfomer_based.transformer_based_network import get_transformer_based_model

from src.network.hybrid_based.Mobile_U_ViT import mobileuvit, mobileuvit_l


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Mobile_U_ViT",
                    choices=["Mobile_U_ViT", "CMUNeXt","CMUNeXt_MKDC", "CMUNeXt_GAG", "CMUNeXt_CMFA", "CMUNeXt_PresenceAux",
                             "CMUNeXt_BoundaryDS", "CMUNeXt_DistanceAux", "CMUNeXt_DualGAG",
                             "CMUNeXt_DualGAG_DistanceAux", "CMUNeXt_SpeckleEnhance", "CMUNet",
                              "AttU_Net", "TransUnet", "R2U_Net", "U_Net",
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=300, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
parser.add_argument('--save_dir', type=str, default="./checkpoint", help='directory to save the best model')
# <=== 新增：是否开启额外数据增强的指令
parser.add_argument('--use_extra_aug', action='store_true', help='Whether to use extra strong data augmentations')
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_MKDC":
        model = cmunext_mkdc(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_GAG":
        model = cmunext_gag(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_CMFA":
        model = cmunext_cmfa(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_PresenceAux":
        model = cmunext_presenceaux(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_BoundaryDS":
        model = cmunext_boundaryds(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_DistanceAux":
        model = cmunext_distanceaux(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_DualGAG":
        model = cmunext_dualgag(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_DualGAG_DistanceAux":
        model = cmunext_dualgag_distanceaux(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_SpeckleEnhance":
        model = cmunext_speckle(num_classes=args.num_classes).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    elif args.model == "Mobile_U_ViT":
        model = mobileuvit(out_channel=args.num_classes).cuda()
    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    return model


def get_criterion(args):
    if args.model == "CMUNeXt_PresenceAux":
        return PresenceAuxLoss().cuda()
    if args.model == "CMUNeXt_BoundaryDS":
        return BoundaryDeepSupervisionLoss().cuda()
    if args.model in {"CMUNeXt_DistanceAux", "CMUNeXt_DualGAG_DistanceAux"}:
        return DistanceAuxLoss().cuda()
    return losses.__dict__['BCEDiceLoss']().cuda()


def forward_with_model(args, model, x, return_aux=True):
    if args.model in {"CMUNeXt_PresenceAux", "CMUNeXt_BoundaryDS", "CMUNeXt_DistanceAux",
                      "CMUNeXt_DualGAG_DistanceAux"}:
        return model(x, return_aux=return_aux)
    return model(x)


def get_seg_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['seg']
    return outputs


def get_loss_tensor(loss_output):
    if isinstance(loss_output, tuple):
        return loss_output[0]
    return loss_output


def get_distance_aux_weight(args, criterion, epoch_num, max_epoch):
    if args.model not in {"CMUNeXt_DistanceAux", "CMUNeXt_DualGAG_DistanceAux"} or not hasattr(criterion, "dist_weight"):
        return None

    base_weight = criterion.dist_weight
    warmup_start = max(5, int(max_epoch * 0.05))
    warmup_end = max(warmup_start + 1, int(max_epoch * 0.15))
    decay_start = max(warmup_end + 1, int(max_epoch * 0.40))
    final_weight = 0.0

    if epoch_num < warmup_start:
        return 0.0
    if epoch_num < warmup_end:
        progress = (epoch_num - warmup_start + 1) / max(1, warmup_end - warmup_start)
        return base_weight * progress
    if epoch_num < decay_start:
        return base_weight

    progress = (epoch_num - decay_start) / max(1, (max_epoch - 1) - decay_start)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return final_weight + (base_weight - final_weight) * cosine


def compute_loss(args, criterion, outputs, label_batch, sampled_batch=None, aux_weight=None):
    if args.model in {"CMUNeXt_DistanceAux", "CMUNeXt_DualGAG_DistanceAux"}:
        distance_target = None
        if sampled_batch is not None and "distance_target" in sampled_batch:
            distance_target = sampled_batch["distance_target"].cuda()
        return criterion(outputs, label_batch, distance_target=distance_target, dist_weight=aux_weight)
    return criterion(outputs, label_batch)


def getDataloader(args, distance_max=None):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = 224

    # <=== 修改：根据参数选择增强策略
    if args.use_extra_aug:
        logging.info("=> Enabled EXTRA strong data augmentation!")
        train_transform = Compose([
            RandomRotate90(p=0.5),
            transforms.Flip(p=0.5),
            # 额外的强增强：弹性形变和网格畸变（对医疗图像很有效）
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            # 额外的强增强：亮度和噪声干扰
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.2),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ])
    else:
        logging.info("=> Using BASIC data augmentation.")
        train_transform = Compose([
            RandomRotate90(p=0.5),
            transforms.Flip(p=0.5),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ])
    # =================================

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                               transform=train_transform, train_file_dir=args.train_file_dir,
                               val_file_dir=args.val_file_dir,
                               use_distance_aux=args.model in {"CMUNeXt_DistanceAux", "CMUNeXt_DualGAG_DistanceAux"},
                               distance_max=distance_max if distance_max is not None else 8.0)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir,
                             use_distance_aux=args.model in {"CMUNeXt_DistanceAux", "CMUNeXt_DualGAG_DistanceAux"},
                             distance_max=distance_max if distance_max is not None else 8.0)
    # <=== 修改 5: 将 print 替换为 logging.info
    logging.info("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader


def main(args):
    # <=== 新增 5: 确保保存目录存在 (使用 exist_ok=True 避免已存在时出错)
    os.makedirs(args.save_dir, exist_ok=True)

    # <=== 新增 6: 配置 logging
    log_file_path = os.path.join(args.save_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # 保存到文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    # =================================

    base_lr = args.base_lr
    model = get_model(args)
    criterion = get_criterion(args)
    distance_max = criterion.max_distance if hasattr(criterion, "max_distance") else None
    trainloader, valloader = getDataloader(args=args, distance_max=distance_max)

    # <=== 修改 6: 将 print 替换为 logging.info
    logging.info("Args: {}".format(args))  # 打印所有参数到日志
    logging.info("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # <=== 修改 7: 将 print 替换为 logging.info
    logging.info("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = args.epoch

    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_iou_history = []

    max_iterations = len(trainloader) * max_epoch

    start_time = time.time()

    for epoch_num in range(max_epoch):
        model.train()
        distance_aux_weight = get_distance_aux_weight(args, criterion, epoch_num, max_epoch)
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_SP': AverageMeter(),
                      'val_HD95': AverageMeter(),
                      'val_ASSD': AverageMeter(),
                      'val_ACC': AverageMeter()}

        # (您修改的部分)
        train_bar = tqdm(trainloader, desc=f"Epoch {epoch_num}/{max_epoch} [Train]")

        for i_batch, sampled_batch in enumerate(train_bar):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

            outputs = forward_with_model(args, model, img_batch)
            seg_logits = get_seg_logits(outputs)

            loss = get_loss_tensor(
                compute_loss(args, criterion, outputs, label_batch, sampled_batch, distance_aux_weight)
            )
            iou, dice, _, _, _, _, _ = iou_score(seg_logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))

            train_bar.set_postfix(loss=avg_meters['loss'].avg, iou=avg_meters['iou'].avg)

        model.eval()
        with torch.no_grad():
            val_bar = tqdm(valloader, desc=f"Epoch {epoch_num}/{max_epoch} [Val  ]")

            for i_batch, sampled_batch in enumerate(val_bar):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                output = forward_with_model(args, model, img_batch)
                seg_logits = get_seg_logits(output)
                loss = get_loss_tensor(
                    compute_loss(args, criterion, output, label_batch, sampled_batch, distance_aux_weight)
                )
                iou, _, SE, PC, F1, SP, ACC = iou_score(seg_logits, label_batch)
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_SP'].update(SP, img_batch.size(0))
                hd95, assd = boundary_scores(seg_logits, label_batch)
                avg_meters['val_HD95'].update(hd95, img_batch.size(0))
                avg_meters['val_ASSD'].update(assd, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

                val_bar.set_postfix(val_loss=avg_meters['val_loss'].avg, val_iou=avg_meters['val_iou'].avg)

        # <=== 修改 8: 将 print 替换为 logging.info
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        if distance_aux_weight is not None:
            logging.info("distance_aux_weight: %.4f", distance_aux_weight)

        logging.info(
            'epoch [%d/%d] (Total time: %s)  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - '
            'val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_HD95 %.4f - val_ASSD %.4f - val_ACC %.4f '
            % (epoch_num, max_epoch, elapsed_str,
               avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
               avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_SP'].avg,
               avg_meters['val_HD95'].avg, avg_meters['val_ASSD'].avg, avg_meters['val_ACC'].avg))
        # <=========================================

        train_loss_history.append(avg_meters['loss'].avg)
        train_iou_history.append(avg_meters['iou'].avg)
        val_loss_history.append(avg_meters['val_loss'].avg)
        val_iou_history.append(avg_meters['val_iou'].avg)

        if avg_meters['val_iou'].avg > best_iou:
            # <=== 修改 9: 使用 args.save_dir 来构建路径
            # 目录已在 main 开头创建，这里无需检查
            save_file_path = os.path.join(args.save_dir, '{}_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_file_path)

            best_iou = avg_meters['val_iou'].avg

            # <=== 修改 10: 将 print 替换为 logging.info
            logging.info(f"=> saved best model to {save_file_path}")

    # <=== 修改 11: 将 print 替换为 logging.info，并使用 args.save_dir
    # 目录已在 main 开头创建，这里无需检查
    logging.info("Saving metric history...")
    np.save(os.path.join(args.save_dir, f'{args.model}_train_loss.npy'), np.array(train_loss_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_train_iou.npy'), np.array(train_iou_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_val_loss.npy'), np.array(val_loss_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_val_iou.npy'), np.array(val_iou_history))

    # <=== 新增 7: 绘制并保存训练曲线图
    logging.info("Saving training curve plots...")
    epochs = range(1, max_epoch + 1)

    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(epochs, train_loss_history, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'r', label='Validation Loss')
    plt.title(f'{args.model} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.model}_loss_plot.png'))
    plt.close()

    # 绘制 IoU 曲线
    plt.figure()
    plt.plot(epochs, train_iou_history, 'b', label='Training IoU')
    plt.plot(epochs, val_iou_history, 'r', label='Validation IoU')
    plt.title(f'{args.model} - Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.model}_iou_plot.png'))
    plt.close()
    # =================================
    # 保存最后一个 epoch 的模型权重
    last_model_path = os.path.join(args.save_dir, '{}_model_last.pth'.format(args.model))
    logging.info(f"=> Saving last epoch model to {last_model_path}")
    torch.save(model.state_dict(), last_model_path)
    # ++++++++++++++++ 添加结束 ++++++++++++++++

    logging.info("Training Finished!")
    return "Training Finished!"


if __name__ == "__main__":
    main(args)



#  cd ~/autodl-tmp/cmu-net
#  libgomp: Invalid value for environment variable OMP_NUM_THREADS：     export OMP_NUM_THREADS=4
#  启动数据增强     --use_extra_aug

# python main.py --model CMUNeXt --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/4.03/busi-CMUNeXt-3-a --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt_DualGAG --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/4.03/busi-CMUNeXt_DualGAG-3-a --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt_DistanceAux --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/4.03/busi-CMUNeXt_DistanceAux-3-a --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt_DualGAG_DistanceAux --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/4.01/busi-CMUNeXt_DualGAG_DistanceAux-3-b --base_lr 0.01 --epoch 300 --batch_size 8 --use_extra_aug

# python main.py --model CMUNeXt_BoundaryDS --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/4.03/busi-CMUNeXt_BoundaryDS-3-a --base_lr 0.01 --epoch 300 --batch_size 8
