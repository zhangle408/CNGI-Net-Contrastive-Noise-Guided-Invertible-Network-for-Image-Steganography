# encoding: utf-8

import functools

import torch
import torch.nn as nn
#from models.module import Harm2d
import torch.nn.functional as F
import math
import numpy as np


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

'''Hnet = UnetGenerator_C(input_nc=opt.channel_secret * opt.num_secret, output_nc=opt.channel_cover * opt.num_cover,
                             num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)'''


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, norm_layer=None):
        super(UnetGenerator, self).__init__()
        self.output_function = nn.Tanh
        '''self.tanh = output_function==nn.Tanh
        if self.tanh:
            self.factor = 10/255
        else:
            self.factor = 1.0'''
        self.factor = 10 / 255
        self.tanh = nn.Tanh
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        # self.relu = nn.ReLU()
        # self.convtran5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bnt5 = norm_layer(512)
        # self.convtran4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bnt4 = norm_layer(256)
        # self.convtran3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bnt3 = norm_layer(128)
        # self.convtran2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bnt2 = norm_layer(64)
        # self.convtran1 = nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
        # 
        # self.drop = nn.Dropout(0.5)

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.bn2(self.conv2(self.leakyrelu(out1)))
        out3 = self.bn3(self.conv3(self.leakyrelu(out2)))
        out4 = self.bn4(self.conv4(self.leakyrelu(out3)))
        out5 = self.conv5(self.leakyrelu(out4))
        # out_5 = self.bnt5(self.convtran5(self.relu(out5)))
        # out_5 = torch.cat([out4, out_5], 1)
        # out_4 = self.bnt4(self.convtran4(self.relu(out_5)))
        # out_4 = self.drop(out_4)
        # out_4 = torch.cat([out3, out_4], 1)
        # out_3 = self.bnt3(self.convtran3(self.relu(out_4)))
        # out_3 = torch.cat([out2, out_3], 1)
        # out_2 = self.bnt2(self.convtran2(self.relu(out_3)))
        # out_2 = torch.cat([out1, out_2], 1)
        # out_1 = self.relu(out_2)
        # out = self.convtran1(out_1)
        

        return out5


