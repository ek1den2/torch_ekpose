import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIなしでmatplotlibを使用
import matplotlib.pyplot as plt

from collections import OrderedDict
from time import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from lib.datasets import datasets, transforms
from lib.config.utils import Logger

# モデルのインポート
from lib.network.networks import get_model


DATA_DIR = './data/'
LOG_DIR = './logs/'
WEIGHTS_DIR = './checkpoints/'


def main():
    parser = argparse.ArgumentParser(description='OpenPose Training Script')
    parser.add_argument('-m', '--model', type=str, default='vgg2016', help='使用モデル名')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='データセットディレクトリ名')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--gpus', type=str, default='0', help='使用するGPUのID（カンマ区切り）')
    parser.add_argument('-l', '--lr', type=float, default=0.0001, help='学習率')
    parser.add_argument('--square_size', type=int, default=160, help='リサイズ後の正方形サイズ')
    parser.add_argument('--loader_workers', type=int, default=8, help='データローダーのワーカー数')

    # モデルの設定
    parser.add_argument('-cw1', '--conv_width1', type=float, default=1.0, help='Convの倍率')
    parser.add_argument('-cw2', '--conv_width2', type=float, default=1.0, help='Conv2の倍率')

    # ログの設定
    parser.add_argument('--training_curve', action='store_true', help='学習曲線を保存するか')
    parser.add_argument('--save_epoch', type=int, default=20, help='モデルの保存間隔（エポック数）')

    # 転移学習の設定
    parser.add_argument('--imagenet_pretrained', action='store_true', help='imagenetで事前学習済みモデルを使用するか')
    parser.add_argument('--pretrained_path', type=str, default=None, help='事前学習済みモデルの重みファイルパス')

    # 最適化関数の設定
    parser.add_argument('--momentum', type=float, default=0.9, help='モメンタム')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='重み減衰')
    parser.add_argument('--nesterov', type=bool, default=True, help='Nesterovの加速（デフォルトで使用）')

    args = parser.parse_args()

    # GPUの設定
    args.pin_memory = False
    if args.gpus:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        args.pin_memory = True
    
    # ログの設定
    TIMESTAMP = "{:%Y%m%d_%H-%M-%S/}".format(datetime.now())
    os.makedirs(LOG_DIR + TIMESTAMP, exist_ok=True)
    os.makedirs(WEIGHTS_DIR + TIMESTAMP, exist_ok=True)

    writer = SummaryWriter(LOG_DIR+TIMESTAMP)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    logfile = os.path.join(LOG_DIR, TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)

    # データのパス設定
    args.train_image_dir = os.path.join(DATA_DIR, args.data_dir, 'images/train')
    args.val_image_dir = os.path.join(DATA_DIR, args.data_dir, 'images/val')
    args.train_annotation_files = [os.path.join(DATA_DIR, args.data_dir, item) for item in ['annotations_train.json']]
    args.val_annotation_file = os.path.join(DATA_DIR, args.data_dir, 'annotations_val.json')

    print("settings:")
    print(vars(args))
    print()



    # 訓練データの読み込み
    preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), probability=0.5),
        transforms.RescaleRelative(),
        transforms.Crop(args.square_size),
        transforms.CenterPad(args.square_size),
    ])

    train_loader, val_loader, train_data, val_data = data_loader(args, preprocess, target_transforms=None)

    
    # modelの読み込み
    model = get_model(
        model_name=args.model,
        pretrained_path=args.pretrained_path,
        imagenet_pretrained=args.imagenet_pretrained,
        conv_width=args.conv_width1,
        conv_width2=args.conv_width2
    )

    model = torch.nn.DataParallel(model).cuda()

    # パラメータ表示（使わん）
    # model_params = 0
    # for param in model.parameters():
    #     model_params += param.numel()
    # print("INFO: Trainable parameters count:", model_params)

    print("INFO: Training Data:", len(train_loader.dataset))
    print("INFO: Validation Data:", len(val_loader.dataset))

    train_loss_history = []
    val_loss_history = []
    stage_names = ['paf1', 'heatmap1', 'paf2', 'heatmap2', 'paf3', 'heatmap3',
               'paf4', 'heatmap4', 'paf5', 'heatmap5', 'paf6', 'heatmap6',
               'max_ht', 'min_ht', 'max_paf', 'min_paf']



    # 事前学習済みモデルを使用する場合
    # 最初の5エポックは特徴マップのレイヤーを凍結して学習
    if args.pretrained_path or args.imagenet_pretrained:
        
        # 特徴マップのレイヤーを凍結
        # print(model.module) # モデルの構造を確認し凍結する層を決定　eg. OpenPose -> model -> (各層)
        for i in range(20):
            for param in model.module.model0[i].parameters():
                param.requires_grad = False


        # 学習が可能なパラメータの取得
        trainable_vars = [param for param in model.parameters() if param.requires_grad]

        # 最適化関数の設定
        # optimizer = torch.optim.SGD(trainable_vars, 
        #                             lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay,
        #                             nesterov=args.nesterov)
        optimizer = torch.optim.Adam(trainable_vars, 
                                    lr=args.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay,
                                    )

        print("\nvvvvvvvvvvv Start WarmUp vvvvvvvvvvv\n")
        for epoch in range(5):
            start_time = time()
            train_loss, train_stage_losses = train(train_loader, model, optimizer, args, epoch)
            val_loss, val_stage_losses = validate(val_loader, model, args, epoch)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            lr = optimizer.param_groups[0]['lr']
            elapsed_time = (time() - start_time) / 60

            print(f'[{epoch+1}] time {elapsed_time:.2f} lr {lr:.6g} train_loss {train_loss:.6f} val_loss {val_loss:.6f}')

        # すべての凍結を解除
        for param in model.parameters():
            param.requires_grad = True
    


    # メイン学習
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    # optimizer = torch.optim.SGD(trainable_vars, 
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)
    optimizer = torch.optim.Adam(trainable_vars, 
                                lr=args.lr,
                                betas=(0.9, 0.999),
                                weight_decay=args.weight_decay,
                                )

    # 学習率のスケジューラ
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=False)
    best_val_loss = np.inf
    

    print("\nvvvvvvvvvvv Start Training vvvvvvvvvvv\n")
    for epoch in range(5 if args.pretrained_path or args.imagenet_pretrained else 0, args.epochs):
        start_time = time()
        train_loss, train_stage_losses = train(train_loader, model, optimizer, args, epoch)
        val_loss, val_stage_losses = validate(val_loader, model, args, epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # 学習率の更新
        lr_scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        elapsed_time = (time() - start_time) / 60
        writer.add_scalar("LearningRate", lr, epoch + 1)


        print(f'[{epoch+1}] time {elapsed_time:.2f} lr {lr:.6g} train_loss {train_loss:.6f} val_loss {val_loss:.6f}')
        
        # 一定間隔でcheckpointを保存
        if (epoch+1) % args.save_epoch == 0:
            chk_path = os.path.join(WEIGHTS_DIR, TIMESTAMP, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), chk_path)
            print(f"save checkpoint: epoch_{epoch+1}.pth")

        # bestモデルの保存
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        best_chk_path = os.path.join(WEIGHTS_DIR, TIMESTAMP, 'best_epoch.pth')
        if is_best and (epoch+1) > 5:
            torch.save(model.state_dict(), best_chk_path)
            print("save best checkpoint")

        # TensorBoard
        writer.add_scalars("Loss", {
            'train_loss': train_loss,
            'val_loss': val_loss
        }, epoch + 1)

        for name, train_stage_loss, val_stage_loss in zip(stage_names, train_stage_losses, val_stage_losses):
            writer.add_scalars(f'{name}', {
                'train': train_stage_loss,
                'val': val_stage_loss
            }, epoch + 1)

        # 学習曲線
        if args.training_curve and epoch+1 > 3:
            plt.figure()
            epoch_x = np.arange(3, len(train_loss_history)) + 1
            plt.plot(epoch_x, train_loss_history[3:], color='blue', label='train_loss')
            plt.plot(epoch_x, val_loss_history[3:], color='orange', label='val_loss')
            plt.legend(['tain_loss', 'val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xlim(3, epoch+1)
            plt.savefig(os.path.join(LOG_DIR, TIMESTAMP, 'training_curve.png'))

            plt.close('all')

    print("\n!!!!!!!!!!!!! Finish Training !!!!!!!!!!!!!\n")






def data_loader(args, preprocess, target_transforms):
        """データローダーを作成"""
        # 訓練データ
        print("Loading train dataset...")
        train_datas = [datasets.CustomKeypoints(
            root=args.train_image_dir,
            annFile=item,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=target_transforms,
            n_images= None,
            input_x=args.square_size,
            input_y=args.square_size,
        ) for item in args.train_annotation_files]

        train_data = ConcatDataset(train_datas)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_workers,
            pin_memory=args.pin_memory,
            drop_last=False
        )

        # 検証データ
        print("Loading val dataset...")
        val_data = datasets.CustomKeypoints(
            root=args.val_image_dir,
            annFile=args.val_annotation_file,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=target_transforms,
            n_images=None,
            input_x=args.square_size,
            input_y=args.square_size,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.loader_workers,
            pin_memory=args.pin_memory,
            drop_last=False
        )

        return train_loader, val_loader, train_data, val_data

def build_names():
    """損失関数の名前を作成"""
    
    names = []
    for stage in range(1, 7):
        for l in range(1, 3):
            names.append(f'loss_stage{stage}_L{l}')
    
    return names

def get_loss(saved_for_loss, heat_temp, vec_temp, args):
    """損失の計算"""
    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='sum').cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j]
        pred2 = saved_for_loss[2 * j + 1] 

        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_temp)

        total_loss += loss1 + loss2

        # ログの保存
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()
    
    # lossをバッチサイズで割る
    total_loss = total_loss / args.batch_size

    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log

def train(train_loader, model, optimizer, args, epoch):
    """訓練"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()

    # モデルを訓練モードに設定
    model.train()

    end = time()
    for i, (img, heatmap_target, paf_target) in enumerate(tqdm(train_loader, desc="train", leave=False)):
        # データの読み込み時間を計測
        data_time.update(time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()

        # モデルの出力（ロスだけ格納）
        _, saved_for_loss = model(img)

        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target, args)

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        
        losses.update(total_loss.item(), img.size(0))


        # 勾配計算、最適化関数
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 経過時間
        batch_time.update(time() - end)
        end = time()

    
    stage_losses = []
    for name, value in meter_dict.items():
        stage_losses.append(value.avg)

    return losses.avg, stage_losses

def validate(val_loader, model, args, epoch):
    """検証"""
    losses = AverageMeter()

    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()

    # モデルを評価モードに設定
    model.eval()

    for i, (img, heatmap_target, paf_target) in enumerate(tqdm(val_loader, desc="val", leave=False)):

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()

        # モデルの出力（ロスだけ格納）
        _, saved_for_loss = model(img)

        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target, args)

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))

        losses.update(total_loss.item(), img.size(0))

    stage_losses = []
    for name, value in meter_dict.items():
        stage_losses.append(value.avg)
    
    return losses.avg, stage_losses


class AverageMeter(object):
    """直近の値、合計値、カウント、平均を保持・計算するクラス"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()