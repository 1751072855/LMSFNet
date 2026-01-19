# -*- coding: utf-8 -*-
"""

import numbers
import os
import random
from datetime import datetime
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from Src.utils.Dataloader import get_loader, test_dataset
from utils.utils import clip_gradient
from lib.pvtv2 import pvt_v2_b2



# ========================== structure_loss ==========================
def structure_loss(pred, mask):
    a = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 1e-8
    b = torch.abs(F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask) + 1e-8
    c = torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask) + 1e-8
    d = torch.abs(F.avg_pool2d(mask, kernel_size=27, stride=1, padding=13) - mask) + 1e-8
    e = torch.abs(F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10) - mask) + 1e-8
    alph = 1.75

    fall = (
            a ** (1.0 / (1 - alph))
            + b ** (1.0 / (1 - alph))
            + c ** (1.0 / (1 - alph))
            + d ** (1.0 / (1 - alph))
            + e ** (1.0 / (1 - alph))
    )
    fall += 1e-8
    a1 = ((a ** (1.0 / (1 - alph)) / fall) ** alph) * a
    b1 = ((b ** (1.0 / (1 - alph)) / fall) ** alph) * b
    c1 = ((c ** (1.0 / (1 - alph)) / fall) ** alph) * c
    d1 = ((d ** (1.0 / (1 - alph)) / fall) ** alph) * d
    e1 = ((e ** (1.0 / (1 - alph)) / fall) ** alph) * e

    weight = 1 + 5 * (a1 + b1 + c1 + d1 + e1)

    dwbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    dwbce = (weight * dwbce).mean(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    dwiou = 1 - (inter + 1e-8) / (union - inter + 1e-8)

    return (dwbce + dwiou).mean()


# ========================== train ==========================
def train(train_loader, model, optimizer, epoch, save_path, writer, opt):
    global step
    model.train()
    loss_all = 0.0
    epoch_step = 0
    scaler = GradScaler()

    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            images = images.cuda()
            gts = gts.cuda()

            with autocast():
                final_pred = model(images)
                total_loss = structure_loss(final_pred, gts)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, opt.clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_step += 1
            loss_all += total_loss.data

            if i % 40 == 0 or i == len(train_loader) or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(
                    datetime.now(), epoch, opt.epoch, i, len(train_loader), total_loss.item()
                ))

            writer.add_scalar('Loss/total', total_loss.item(), global_step=step)

        loss_all /= epoch_step
        print('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        # 每 20 epoch 存一个 checkpoint
        if epoch % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f'PVTv2_MSSEP_epoch_{epoch}_loss_{loss_all:.4f}.pth')
            )

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f'PVTv2_MSSEP_interrupt_epoch_{epoch}.pth')
        )
        print('Save checkpoints successfully!')
        raise


# ========================== val ==========================
def val(test_loader, model, epoch, save_path, writer, best_metrics):
    model.eval()
    total_mae = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for i in range(test_loader.size):
            count += 1
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.unsqueeze(0).cuda()
            image = image.squeeze(1)

            final_pred = model(image)

            res = F.interpolate(final_pred, size=gt.shape, mode='bilinear', align_corners=False)
            res_sig = res.sigmoid().squeeze().cpu().numpy()
            res_sig = (res_sig - res_sig.min()) / (res_sig.max() - res_sig.min() + 1e-8)

            # MAE
            mae = np.mean(np.abs(res_sig - gt))
            total_mae += mae

            # IoU
            res_bin = (res_sig > 0.5).astype(np.float32)
            gt_bin = (gt > 0.5).astype(np.float32)
            intersection = np.sum(res_bin * gt_bin)
            union = np.sum(res_bin) + np.sum(gt_bin) - intersection
            iou = (intersection + 1e-8) / (union + 1e-8)
            total_iou += iou

            # Dice
            dice = (2 * intersection + 1e-8) / (np.sum(res_bin) + np.sum(gt_bin) + 1e-8)
            total_dice += dice

        epoch_mae = total_mae / count
        epoch_iou = total_iou / count
        epoch_dice = total_dice / count

        print(
            f'\n[Val Result] Epoch: {epoch:03d} | MAE: {epoch_mae:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}')
        print(
            f'[Best Result] Epoch: {best_metrics["epoch"]:03d} | MAE: {best_metrics["mae"]:.4f} | IoU: {best_metrics["iou"]:.4f} | Dice: {best_metrics["dice"]:.4f}\n')

        writer.add_scalars('Val_Metrics',
                           {'MAE': epoch_mae, 'IoU': epoch_iou, 'Dice': epoch_dice},
                           global_step=epoch)

        current_score = 0.6 * epoch_iou + 0.3 * epoch_dice + 0.1 * (1 - epoch_mae)
        best_score = best_metrics["score"]

        if current_score > best_score:
            best_metrics["epoch"] = epoch
            best_metrics["mae"] = epoch_mae
            best_metrics["iou"] = epoch_iou
            best_metrics["dice"] = epoch_dice
            best_metrics["score"] = current_score
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path,
                    f'LMSFNet_best_epoch_{epoch}_iou_{epoch_iou:.4f}_dice_{epoch_dice:.4f}.pth'
                )
            )
            print(f'[Val Info] Save BEST model at Epoch:{epoch}, Score:{current_score:.4f}')

    torch.cuda.empty_cache()
    return best_metrics, current_score


# ========================== main ==========================
if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=8)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--train_root', type=str,
                        default=r'')
    parser.add_argument('--val_root', type=str,
                        default=r'')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--save_path', type=str,
                        default=r'')
    parser.add_argument('--pvt_weight', type=str,
                        default=r'')
    parser.add_argument('--use_mssep', type=int, default=1 )

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = False

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.save_path, 'train.log'),
        format='[%(asctime)s-%(levelname)s]: %(message)s',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info('>>> Training Configuration:')
    logging.info(f'Parameters: {opt}')
    print(f'>>> Training Configuration: {opt}')

    use_mssep = bool(opt.use_mssep)

    model = LMSFNet(
        pretrained_pvt_path=opt.pvt_weight,
        out_channels=1,
        fuse_channels=64,
        use_mssep=use_mssep
    ).cuda()
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-5
    )

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print(f'Loaded checkpoint from: {opt.load}')

    print('Loading dataset...')
    train_loader = get_loader(
        image_root=opt.train_root + 'Image/',
        gt_root=opt.train_root + 'GT/',
        edge_root=opt.train_root + 'Edge/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=4
    )
    val_loader = test_dataset(
        image_root=opt.val_root + 'Image/',
        gt_root=opt.val_root + 'GT/',
        testsize=opt.trainsize
    )

    writer = SummaryWriter(opt.save_path + 'summary')
    step = 0

    best_metrics = {
        "epoch": 0,
        "mae": float('inf'),
        "iou": 0.0,
        "dice": 0.0,
        "score": 0.0
    }

    # warmup + cosine
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.01,
        total_iters=5
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=50,
        eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5]
    )

    early_stop_count = 0
    best_score = 0.0

    print(">>> Starting model ablation training...")
    for epoch in range(1, opt.epoch + 1):
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
        logging.info(f'[Epoch {epoch}] Current learning rate: {current_lr:.8f}')

        train(train_loader, model, optimizer, epoch, opt.save_path, writer, opt)

        best_metrics, current_score = val(val_loader, model, epoch, opt.save_path, writer, best_metrics)

        if current_score > best_score:
            best_score = current_score
            early_stop_count = 0
        else:
            early_stop_count += 1
            logging.info(f'[Early Stop] No improvement for {early_stop_count}/{opt.early_stop_patience} epochs')

        if early_stop_count >= opt.early_stop_patience:
            print(f'\n>>> Early stopping triggered! Best epoch: {best_metrics["epoch"]}')
            print(
                f'Best performance: MAE={best_metrics["mae"]:.4f}, IoU={best_metrics["iou"]:.4f}, Dice={best_metrics["dice"]:.4f}'
            )
            logging.info(f'[Early Stop] Stopped at epoch {epoch}. Best epoch: {best_metrics["epoch"]}')
            break

    torch.save(
        model.state_dict(),
        os.path.join(opt.save_path, f'Model_final_epoch_{epoch}.pth')
    )
    print(">>> Training completed!")
    logging.info(">>> Training completed successfully!")
    writer.close()