# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random

import torch
from torch.autograd import Variable
from models.unet import UNet
import glob
from models.xnet import XNet
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
from utils.metrics import compute_metrics

# 权重地址
train_weights = r'E:\论文\stoneseg(1)\trainingrecords\checkpoint\mydata_unet_dy_Lovasz_adam_0.0001\mydata_unet_dy_Lovasz_adam_0.0001_best_epoch_185_0.965.pkl'

model_name = "unet_dy"

# 选择网络模型
# 模型声明
if model_name == "unet":
    net = UNet(num_classes=2, in_channels=1, is_esa=False, is_grid=False, is_dy=False)
elif model_name == 'xnet':
    net = XNet(in_channels=1, num_classes=2)
elif model_name == 'unet_dy':
    net = UNet(num_classes=2, in_channels=1, is_esa=False, is_grid=False, is_dy=True)
elif model_name == 'unet_dy_esa':
    net = UNet(num_classes=2, in_channels=1, is_esa=True, is_grid=False, is_dy=True)
elif model_name == 'unet_esa':
    net = UNet(num_classes=2, in_channels=1, is_esa=True, is_grid=False, is_dy=False)

ckpt = torch.load(train_weights, map_location=torch.device('cpu'))
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    new_state_dict[k] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(new_state_dict)
net.eval()

pre_data_path = r'E:\test\image'


dst_path = '\\'.join(train_weights.split('\\')[:-1]).replace('checkpoint', 'pred_' + model_name)
os.makedirs(dst_path, exist_ok=True)


image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))
# 验证
miou_total, mdsc_total, ac_total, mpc_total, mse_total \
    , msp_total, mf1_total = 0, 0, 0, 0, 0, 0, 0
nums = len(image_list)

image_list1 = []
for file in os.listdir(r"E:\test\image"):
    image_list1.append(os.path.join(pre_data_path, file))

predictions_all = []
labels_all = []

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("\\")[-1]
        labels = cv2.imread(image.replace('image', 'mask'), 0)
        labels[labels == 255] = 1

        image = Image.open(image).convert('L')
        image = transforms.ToTensor()(image)
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        outputs = net(image)
        if isinstance(outputs, list):
            # 若使用deep supervision，用最后一个输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = labels.astype(np.int)
        predictions_all.append(predictions)
        labels_all.append(labels)
        if isinstance(outputs, list):
            outputs = outputs[-1].squeeze(0)    # [2, 224, 224]

        else:
            outputs = outputs.squeeze(0)    # [2, 224, 224]
        mask = torch.max(outputs, 0)[1].cpu().numpy()

        new_mask = np.zeros((512, 512, 3))

        new_mask[:, :, 2][np.where(mask == 1)] = 255
        new_mask[:, :, 1][np.where(mask == 1)] = 255
        new_mask[:, :, 0][np.where(mask == 1)] = 255

        base_path = r"E:\test\pred_unet_dy"
        file_name = str(name)  # 确保name是字符串类型，但在这个例子中如果name已经是字符串则不需要转换
        full_path = os.path.join(base_path, file_name)
        cv2.imwrite(full_path, new_mask)
        base_image = cv2.imread(image_list1[i-1])
        if new_mask.ndim == 3 and new_mask.shape[2] == 3:
            new_mask = (new_mask[:, :, 0] > 0).astype(np.uint8) * 255
        if new_mask.ndim == 2:
            new_mask = np.dstack((new_mask, new_mask, new_mask))  # 将单通道掩码扩展为三通道
        # 如果 new_mask 是浮点数类型（0.0 到 1.0），则需要先将其转换回整数类型
        if new_mask.dtype == np.float32:
            new_mask = (new_mask * 255).astype(np.uint8)
            # 现在 new_mask 和 base_image 应该都是 np.uint8 类型
        # 并且可以安全地传递给 cv2.addWeighted()
        colored_mask = np.zeros_like(base_image)

        # 定义颜色 (B, G, R)，例如：
        color_1 = [0, 255, 0]  # 绿色
        color_2 = [0, 0, 255]  # 红色
        color_3 = [0, 255, 255] # 黄色
        color_4 = [255, 178, 102] # 橙色
        # 将掩码中值为 255 的部分映射为绿色
        colored_mask[new_mask[:, :, 0] == 255] = color_4
        # 混合 base_image 和 colored_mask
        result = cv2.addWeighted(base_image, 0.8, colored_mask, 0.2, 0)
        # 显示或保存结果图像
        cv2.imwrite(full_path, result)
        # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   num_classes=2)
    print(
        'Testing: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
            miou, mdsc, mpc, ac, mse, msp, mf1
        ))



