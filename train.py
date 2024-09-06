#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from models.xnet import XNet
from utils import loss_function
from datasets import mydataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import numpy as np
from utils.metrics import compute_metrics

# 命令行参数设置
parser = argparse.ArgumentParser()
# pytorch

parser.add_argument('--model', type=str, default='unet_dy_esa')    # unet, xnet, unet_dy, unet_dy_esa, unet_esa
parser.add_argument('--dataset', type=str, default='mydata', choices=['mydata'])
parser.add_argument('--loss', type=str, default='Lovasz', choices=['Dice'])
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'])
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
parser.add_argument('--parallel', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--num_workers', type=int, default=0, choices=list(range(17)))
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)   # 初始学习率， Adam
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--print_frequency', type=int, default=4)
parser.add_argument('--save_frequency', type=int, default=1)
args = parser.parse_args()

# 其他准备
BASE_PATH = r'H:\stoneseg\trainingrecords'
format = '{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.loss, args.optimizer, args.lr)


checkpoint_path_prefix = os.path.join(BASE_PATH, 'checkpoint', format)
os.makedirs(checkpoint_path_prefix, exist_ok=True)

writer = SummaryWriter(log_dir=checkpoint_path_prefix, flush_secs=30)


DEVICE = 'cuda'

# 加载数据
print('Loading data...')
# 选择数据集
if args.dataset == 'mydata':
    dataset = mydataset.Mydataset
else:
    print('数据集异常')
    pass

# 对image和mask进行resize
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.ToTensor()])

# noisy_chaos可以设置噪声率和噪声类型

if args.dataset == 'mydata':
    train_data = dataset(mode='train', transform=transform, target_transform=target_transform,
                         BASE_PATH=r"H:\stoneseg\data\train")
    val_data = dataset(mode='val', transform=transform, target_transform=target_transform,
                       BASE_PATH=r"H:\stoneseg\data\test")
else:
    pass


train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

print('Create model...')
# 选择网络模型
in_channel = dataset.CHANNELS_NUM
if args.model == "unet":
    net = UNet(num_classes=2, in_channels=in_channel, is_esa=False, is_grid=False, is_dy=False)
elif args.model == 'xnet':
    net = XNet(in_channels=in_channel, num_classes=2)
elif args.model == 'unet_att':
    net = UNet(num_classes=2, in_channels=in_channel, is_esa=False, is_grid=True, is_dy=False)
elif args.model == 'unet_dy':
    net = UNet(num_classes=2, in_channels=in_channel, is_esa=False, is_grid=False, is_dy=True)
elif args.model == 'unet_dy_esa':
    net = UNet(num_classes=2, in_channels=in_channel, is_esa=True, is_grid=False, is_dy=True)
elif args.model == 'unet_esa':
    net = UNet(num_classes=2, in_channels=in_channel, is_esa=True, is_grid=False, is_dy=False)


# 设置优化方法和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# 选择损失函数
if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'Dice':
    criterion = loss_function.MultiClassDiceLoss(dataset.NUM_CLASSES)
elif args.loss == 'Lovasz':
    criterion = loss_function.LovaszLoss()


print('<================== Parameters ==================>')
print('model: {}'.format(net))
print('dataset: {}(training={}, validation={})'.format(train_data, len(train_data), len(val_data)))
print('batch_size: {}'.format(args.batch_size))
print('batch_num: {}'.format(len(train_loader)))
print('epoch: {}'.format(args.epoch))
print('loss_function: {}'.format(criterion))
print('optimizer: {}'.format(optimizer))
print('<================================================>')


net = net.to(DEVICE)

start_epoch = 0
temp = 0
# 加载模型
if args.checkpoint is not None:
    checkpoint_data = torch.load(args.checkpoint)
    print('**** Load model and optimizer data from {} ****'.format(args.checkpoint))

    # 加载模型和优化器的数据
    net.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # 加载上次训练的最后一个epoch和打印的最后一个temp，这里作为起点，需要在之前的基础上加1
    start_epoch = checkpoint_data['epoch'] + 1
    temp = checkpoint_data['temp'] + 1

    args.epoch += start_epoch

# 训练与验证的过程
print('Start training...')
# 设置早停初始值
best_dice = 0.0
for epoch in range(start_epoch, args.epoch):
    # 训练
    loss_all = []
    predictions_all = []
    labels_all = []
    print('-------------------------------------- Training {} --------------------------------------'.format(epoch + 1))
    net.train()
    for index, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)   #
        loss = 0
        # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
        if isinstance(outputs, list):
            for out in outputs:
                loss += criterion(out, labels.long())
            loss /= len(outputs)
        else:
            loss = criterion(outputs, labels.long())
        # 计算在该批次上的平均损失函数
        loss /= inputs.size(0)

        # 更新网络参数
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

        if isinstance(outputs, list):
            # 若使用deep supervision，用最后的输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int8)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int8)

        labels = labels.cpu().numpy().astype(np.int8)

        predictions_all.append(predictions)
        labels_all.append(labels)

        if (index + 1) % args.print_frequency == 0:
            # 计算打印间隔的平均损失函数
            avg_loss = np.mean(loss_all)
            loss_all = []
            writer.add_scalar('train/loss', avg_loss, temp)

            temp += 1

            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch + 1, args.epoch, index + 1, len(train_loader), avg_loss))

    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)

    writer.add_scalars('train/metrics', dict(miou=miou, mdsc=mdsc, mpc=mpc, ac=ac, mse=mse, msp=msp, mf1=mf1), epoch)


    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 验证
    loss_all = []
    predictions_all = []
    labels_all = []

    print('-------------------------------------- Validation {} ------------------------------------'.format(epoch + 1))

    net.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = net(inputs)

            loss = 0
            # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
            if isinstance(outputs, list):
                for out in outputs:
                    loss += criterion(out, labels.long())
                loss /= len(outputs)
            else:
                loss = criterion(outputs, labels.long())
            # 计算在该批次上的平均损失函数
            loss /= inputs.size(0)

            loss_all.append(loss.item())

            if isinstance(outputs, list):
                # 若使用deep supervision，用最后一个输出来进行预测
                predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int8)
            else:
                # 将概率最大的类别作为预测的类别
                predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int8)
            labels = labels.cpu().numpy().astype(np.int8)

            predictions_all.append(predictions)
            labels_all.append(labels)

    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)
    avg_loss = np.mean(loss_all)

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalars('val/metrics', dict(miou=miou, mdsc=mdsc, mpc=mpc, ac=ac, mse=mse, msp=msp, mf1=mf1), epoch)

    # 绘制每个类别的IoU
    temp_dict = {'miou': miou}
    for i in range(dataset.NUM_CLASSES):
        temp_dict['class{}'.format(i)] = iou[i]
    writer.add_scalars('val/class_iou', temp_dict, epoch)

    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 保存模型参数和优化器参数
    if mdsc >= best_dice:
        best_dice = mdsc
        checkpoint_path = '{}_best_epoch_{}_{}.pkl'.format(format, str(epoch), str(np.round(mdsc, 3)))
        # save_checkpoint_path = checkpoint_path_prefix + '/' + checkpoint_path
        save_checkpoint_path = os.path.join(checkpoint_path_prefix, checkpoint_path)
        torch.save({
            'is_parallel': args.parallel,
            'epoch': epoch,
            'temp': temp,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            save_checkpoint_path)
        print('Save model at {}.'.format(save_checkpoint_path))
