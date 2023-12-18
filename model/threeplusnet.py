# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, padding_mode = 'reflect'),
                                     nn.BatchNorm2d(out_size),
                                     nn.SELU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, padding_mode = 'reflect'),
                                     nn.SELU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

'''
    UNet 3+
'''
class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.SELU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.SELU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.SELU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.SELU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.SELU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.SELU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.SELU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.SELU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.SELU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.SELU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.SELU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.SELU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.SELU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.SELU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1, padding_mode = 'reflect')

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1
        # return F.sigmoid(d1)


class UNet_3PlusMemOpt(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.SELU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.SELU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.SELU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.SELU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.SELU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.SELU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.SELU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.SELU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.SELU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.SELU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.SELU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.SELU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.SELU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.SELU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.SELU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.SELU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.SELU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1, padding_mode = 'reflect')

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        model_device = inputs.device
        current_device = torch.cuda.current_device()

        print ('forward start:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64
        del inputs
        torch.cuda.empty_cache()

        print ('del inputs:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2 = self.maxpool1(h1)
        h1_cpu = h1.to('cpu')
        del h1
        torch.cuda.empty_cache()

        print ('del h1:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h2_cpu = h2.to('cpu')
        del h2
        torch.cuda.empty_cache()

        print ('del h2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h3_cpu = h3.to('cpu')
        del h3
        torch.cuda.empty_cache()

        print ('del h3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h4 = self.conv4(h4)  # hå4->40*40*512
        h5 = self.maxpool4(h4)

        h4_cpu = h4.to('cpu')
        del h4
        torch.cuda.empty_cache()

        print ('del h4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5 = self.conv5(h5)  # h5->20*20*1024

        print ('conv hd5:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        del h5
        torch.cuda.empty_cache()

        print ('del h5:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5_cpu = hd5.to('cpu')
        del hd5
        torch.cuda.empty_cache()

        print ('del hd5:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        ## -------------Decoder-------------
        h1 = h1_cpu.to(model_device)
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))

        print ('h1_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h1_PT_hd4_cpu = h1_PT_hd4.to('cpu')
        del h1
        del h1_PT_hd4
        torch.cuda.empty_cache()

        print ('del h1_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2 = h2_cpu.to(model_device)
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))

        print ('h2_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2_PT_hd4_cpu = h2_PT_hd4.to('cpu')
        del h2
        del h2_PT_hd4
        torch.cuda.empty_cache()

        print ('del h2_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h3 = h3_cpu.to(model_device)
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))

        print ('h3_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h3_PT_hd4_cpu = h3_PT_hd4.to('cpu')
        del h3
        del h3_PT_hd4
        torch.cuda.empty_cache()

        print ('del h3_PT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h4 = h4_cpu.to(model_device)
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))

        print ('h4_Cat_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h4_Cat_hd4_cpu = h4_Cat_hd4.to('cpu')
        del h4
        del h4_cpu
        del h4_Cat_hd4
        torch.cuda.empty_cache()

        print ('del h4_Cat_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5 = hd5_cpu.to(model_device)
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        print ('hd5_UT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5_UT_hd4_cpu = hd5_UT_hd4.to('cpu')
        del hd5
        del hd5_UT_hd4
        torch.cuda.empty_cache()

        print ('del hd5_UT_hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd4_cat_cpu = torch.cat((h1_PT_hd4_cpu, h2_PT_hd4_cpu, h3_PT_hd4_cpu, h4_Cat_hd4_cpu, hd5_UT_hd4_cpu), 1)
        hd4_cat = hd4_cat_cpu.to(model_device)

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(hd4_cat))) # hd4->40*40*UpChannels

        print ('hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd4_cpu = hd4.to('cpu')
        del hd4
        del h1_PT_hd4_cpu
        del h2_PT_hd4_cpu
        del h3_PT_hd4_cpu
        del h4_Cat_hd4_cpu
        del hd5_UT_hd4_cpu
        torch.cuda.empty_cache()

        print ('del hd4:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h1 = h1_cpu.to(model_device)
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h1_PT_hd3_cpu = h1_PT_hd3.to('cpu')
        del h1
        del h1_PT_hd3
        torch.cuda.empty_cache()
        
        print ('del h1_PT_hd3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2 = h2_cpu.to(model_device)
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h2_PT_hd3_cpu = h2_PT_hd3.to('cpu')
        del h2
        del h2_PT_hd3
        torch.cuda.empty_cache()

        print ('del h2_PT_hd3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h3 = h3_cpu.to(model_device)
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        h3_Cat_hd3_cpu = h3_Cat_hd3.to('cpu')
        del h3
        del h3_cpu
        del h3_Cat_hd3
        torch.cuda.empty_cache()

        print ('del h3_Cat_hd3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd4 = hd4_cpu.to(model_device)
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd4_UT_hd3_cpu = hd4_UT_hd3.to('cpu')
        del hd4
        del hd4_UT_hd3
        torch.cuda.empty_cache()

        print ('del hd4_UT_hd3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5 = hd5_cpu.to(model_device)
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd5_UT_hd3_cpu = hd5_UT_hd3.to('cpu')
        del hd5
        del hd5_UT_hd3
        torch.cuda.empty_cache()

        print ('del hd5_UT_hd3:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd3_cat_cpu = torch.cat((h1_PT_hd3_cpu, h2_PT_hd3_cpu, h3_Cat_hd3_cpu, hd4_UT_hd3_cpu, hd5_UT_hd3_cpu), 1)
        hd3_cat = hd3_cat_cpu.to(model_device)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(hd3_cat))) # hd3->80*80*UpChannels
        hd3_cpu = hd3.to('cpu')
        del hd3
        del hd3_cat
        del hd3_cat_cpu
        del h1_PT_hd3_cpu
        del h2_PT_hd3_cpu
        del h3_Cat_hd3_cpu
        del hd4_UT_hd3_cpu
        del hd5_UT_hd3_cpu
        torch.cuda.empty_cache()

        print ('hd3_cpu:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h1 = h1_cpu.to(model_device)
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h1_PT_hd2_cpu = h1_PT_hd2.to('cpu')
        del h1
        del h1_PT_hd2
        torch.cuda.empty_cache()

        print ('del h1_PT_hd2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h2 = h2_cpu.to(model_device)
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        h2_Cat_hd2_cpu = h2_Cat_hd2.to('cpu')
        del h2
        del h2_cpu
        del h2_Cat_hd2
        torch.cuda.empty_cache()

        print ('del h2_Cat_hd2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd3 = hd3_cpu.to(model_device)
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd3_UT_hd2_cpu = hd3_UT_hd2.to('cpu')
        del hd3
        del hd3_UT_hd2
        torch.cuda.empty_cache()

        print ('del hd3_UT_hd2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd4 = hd4_cpu.to(model_device)
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd4_UT_hd2_cpu = hd4_UT_hd2.to('cpu')
        del hd4
        del hd4_UT_hd2
        torch.cuda.empty_cache()

        print ('del hd4_UT_hd2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5 = hd5_cpu.to(model_device)
        hd5_ut = self.hd5_UT_hd2(hd5)
        del hd5
        torch.cuda.empty_cache()

        print ('hd5_ut:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5_ut_conv = self.hd5_UT_hd2_conv(hd5_ut)
        del hd5_ut
        torch.cuda.empty_cache()

        print ('hd5_ut_conv:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd5_UT_hd2_cpu = hd5_UT_hd2.to('cpu')
        del hd5
        del hd5_UT_hd2
        torch.cuda.empty_cache()

        print ('del hd5_UT_hd2:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        hd2_cat_cpu = torch.cat((h1_PT_hd2_cpu, h2_Cat_hd2_cpu, hd3_UT_hd2_cpu, hd4_UT_hd2_cpu, hd5_UT_hd2_cpu), 1)
        hd2_cat = hd2_cat_cpu.to(model_device)

        hd2_conv = self.conv2d_1(hd2_cat)

        torch.cuda.empty_cache()
        print ('hd2_cat:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        # hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(hd2_cat))) # hd2->160*160*UpChannels
        hd2_cpu = hd2.to('cpu')

        del hd2
        del hd2_cat
        del hd2_cat_cpu
        del h1_PT_hd2_cpu
        del h2_Cat_hd2_cpu
        del hd3_UT_hd2_cpu
        del hd4_UT_hd2_cpu
        del hd5_UT_hd2_cpu
        torch.cuda.empty_cache()

        print ('hd2_cpu:')
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        del h1
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        del hd2
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        del hd3
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        del hd4
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        del hd5
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        del h1_Cat_hd1
        del hd2_UT_hd1
        del hd3_UT_hd1
        del hd4_UT_hd1
        del hd5_UT_hd1
        torch.cuda.empty_cache()

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        del hd1
        return d1
        # return F.sigmoid(d1)
