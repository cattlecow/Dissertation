# -*- coding: utf-8 -*-
import os
import pandas as pd
from matplotlib import pyplot as plt

'''
绘制loss图
'''


def loss_visualize(epoch_loss_unet_dy, value_loss_unet_dy, epoch_loss_unet_dy_esa, value_loss_unet_dy_esa
                   , epoch_loss_unet, value_loss_unet, epoch_loss_xnet, value_loss_xnet, epoch_loss_unet_esa, value_loss_unet_esa):
    plt.style.use('tableau-colorblind10')
    # print(plt.style.available)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Training Loss Curve")

    plt.plot(epoch_loss_unet, value_loss_unet, label='Unet', color='c', linestyle='-')
    plt.plot(epoch_loss_xnet, value_loss_xnet, label='XNet', color='g',
             linestyle='-')
    plt.plot(epoch_loss_unet_dy, value_loss_unet_dy, label='Unet_Dy', color='b', linestyle='-')
    plt.plot(epoch_loss_unet_esa, value_loss_unet_esa, label='Unet_ESA', color='k', linestyle='-')

    plt.plot(epoch_loss_unet_dy_esa, value_loss_unet_dy_esa, label='Unet_Dy_ESA', color='m', linestyle='-')



    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(os.path.join(res_dir, r'loss.eps'), dpi=350, format='eps')
    plt.savefig(os.path.join(res_dir, r'loss.png'), dpi=350, format='png')
    plt.savefig(os.path.join(res_dir, r'loss.svg'), dpi=350, format='svg')

    plt.show()


def read_value(train_df):
    epoch = train_df['Step']
    value = train_df['Value']
    return epoch, value


if __name__ == "__main__":
    root_dir = os.getcwd()
    file_dir = os.path.join(root_dir, r'H:\stoneseg\trainingrecords')
    res_dir = os.path.join(root_dir, r'H:\stoneseg\trainingrecords')
    loss_unet = pd.read_csv(os.path.join(file_dir, 'mydata_unet_Lovasz_adam_0.0001.csv'))
    loss_unet_dy = pd.read_csv(os.path.join(file_dir, 'mydata_unet_dy_Lovasz_adam_0.0001.csv'))
    loss_unet_esa = pd.read_csv(os.path.join(file_dir, 'mydata_unet_esa_Lovasz_adam_0.0001.csv'))
    loss_xnet = pd.read_csv(os.path.join(file_dir, 'mydata_xnet_Lovasz_adam_0.0001.csv'))
    loss_unet_dy_esa = pd.read_csv(os.path.join(file_dir, 'mydata_unet_dy_esa_Lovasz_adam_0.0001.csv'))
    epoch_loss_unet_dy, value_loss_unet_dy = read_value(loss_unet_dy)
    epoch_loss_unet_dy_esa, value_loss_unet_dy_esa = read_value(loss_unet_dy_esa)
    epoch_loss_unet_esa, value_loss_unet_esa = read_value(loss_unet_esa)
    epoch_loss_unet, value_loss_unet = read_value(loss_unet)
    epoch_loss_xnet, value_loss_xnet = read_value(loss_xnet)

    loss_visualize(epoch_loss_unet_dy, value_loss_unet_dy, epoch_loss_unet_dy_esa, value_loss_unet_dy_esa
                   , epoch_loss_unet, value_loss_unet, epoch_loss_xnet, value_loss_xnet, epoch_loss_unet_esa, value_loss_unet_esa)
