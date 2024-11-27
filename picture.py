import warnings
import sys
import math
import os
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import tqdm
# import cv2
from model import *
from imp_subnet import *
import config_s1 as c
from os.path import join
import dataset_s1 as datasets
import modules.module_util as mutil
import modules.Unet_common as common

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]




def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


net1 = Model_1()
#net2 = Model_2()
#net3 = ImpMapBlock()
net1.cuda()
#net2.cuda()
#net3.cuda()
init_model(net1)
#init_model(net2)
net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
#net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
#net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
#params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
#params_trainable3 = (list(filter(lambda p: p.requires_grad, net3.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
#optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
#optim3 = torch.optim.Adam(params_trainable3, lr=c.lr3, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
#weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
#weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, c.weight_step, gamma=c.gamma)
dwt = common.DWT()
iwt = common.IWT()

if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
    #load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2)
   # if c.PRETRAIN_PATH_3 is not None:
        #load(c.PRETRAIN_PATH_3 + c.suffix_pretrain_3 + '_3.pt', net3, optim3)


with torch.no_grad():
    net1.eval()
    #net2.eval()
    #net3.eval()
    import time
    start = time.time()
    #for i, x in enumerate(datasets.testloader):
    for i, ((secret_1, secret_target), (cover, cover_target)) in enumerate(datasets.val_loader, 0):
        print(i)
        cover = cover.to(device)
        secret_1 = secret_1.to(device)
        #x = x.to(device)
        # cover = x[:x.shape[0] // 3]  # channels = 3
        # secret_1 = x[x.shape[0] // 3: 2 * x.shape[0] // 3]
        # secret_2 = x[2 * x.shape[0] // 3: 3 * x.shape[0] // 3]
        #cover = x[x.shape[0] // 2:]  # channels = 3
        #secret_1 = x[:x.shape[0] // 2]
        #print(x.shape)
        #print(cover.shape)
        #print(secret_1.shape)
        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
       #secret_dwt_2 = dwt(secret_2)
        #print(cover_dwt.shape)
        #print(secret_dwt_1.shape)
        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24
        
        print(input_dwt_1.shape)

        #################
        #    forward1:   #
        #################
        output_dwt_1 = net1(input_dwt_1)  # channels = 24
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

        # get steg1
        output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3

        # #################
        # #    forward2:   #
        # #################
        # if c.use_imp_map:
        #     imp_map = net3(cover, secret_1, output_steg_1)  # channels = 3
        # else:
        #     imp_map = torch.zeros(cover.shape).cuda()
        #
        # imp_map_dwt = dwt(imp_map)  # channels = 12
        # input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)  # 24, without secret2
        # input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 36
        #
        # output_dwt_2 = net2(input_dwt_2)  # channels = 36
        # output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        # output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24
        #
        # # get steg2
        # output_steg_2 = iwt(output_steg_dwt_2).to(device)  # channels = 3
        #
        # #################
        # #   backward2:   #
        # #################
        #
        # output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
        # output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24
        #
        # output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36
        #
        # rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36
        #
        # rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        # rev_secret_dwt_2 = rev_dwt_2.narrow(1, output_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        #
        # rev_steg_1 = iwt(rev_steg_dwt_1).to(device)  # channels = 3
        # rev_secret_2 = iwt(rev_secret_dwt_2).to(device)  # channels = 3

        #################
        #   backward1:   #
        #################

        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
        output_rev_dwt_1 = torch.cat((output_steg_dwt_1, output_z_guass_1), 1)  # channels = 24
        
        print(output_rev_dwt_1.shape)
        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

        rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt).to(device)

        print(output_steg_1.is_cuda)
        print(cover.is_cuda)
        cover_gap = output_steg_1 - cover
        secret_gap = rev_secret_1 - secret_1
        cover_gap_10 = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)
        secret_gap_10 = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)
        cover_gap_5 = (cover_gap * 5 + 0.5).clamp_(0.0, 1.0)
        secret_gap_5 = (secret_gap * 5 + 0.5).clamp_(0.0, 1.0)
        cover_gap_1 = (cover_gap * 1 + 0.5).clamp_(0.0, 1.0)
        secret_gap_1 = (secret_gap * 1 + 0.5).clamp_(0.0, 1.0)
        print(output_steg_1.shape)
        print(cover.shape)
        print(cover_gap.shape)

        # vutils.save_image(secret_i, 'secret'+str(i_secret)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)
        # vutils.save_image(rev_secret_i, 'rev_secret'+str(i_secret)+'.png', nrow=bs_secret_times_num_training, padding=0, normalize=True)

        GAP_COVER_PATH = '/home/amax/litong/DeepMIH/test/parisC1S1/num14/'
        GAP_SECRET_PATH = '/home/amax/litong/DeepMIH/test/parisC1S1/num14/'
        
        torchvision.utils.save_image(secret_gap_10, GAP_SECRET_PATH+'secret_gap10_'+ '.png', nrow = 1,
                          padding=0, normalize=True)
        torchvision.utils.save_image(secret_gap_5, GAP_SECRET_PATH+'secret_gap5_' + '.png', nrow = 1,
                          padding=0, normalize=True)
        torchvision.utils.save_image(secret_gap_1, GAP_SECRET_PATH+'secret_gap1_' + '.png', nrow = 1,
                          padding=0, normalize=True)

        torchvision.utils.save_image( cover_gap_10, GAP_COVER_PATH + 'cover_gap10_' + '.png',
                                     padding=0, normalize=True)
        torchvision.utils.save_image( cover_gap_5, GAP_COVER_PATH + 'cover_gap5_' + '.png',
                                     padding=0, normalize=True)
        torchvision.utils.save_image( cover_gap_1, GAP_COVER_PATH + 'cover_gap1_' + '.png',
                                     padding=0, normalize=True)
        
        

        torchvision.utils.save_image(cover, GAP_COVER_PATH + 'cover.png')
        torchvision.utils.save_image(secret_1, GAP_SECRET_PATH + 'secret.png')
        torchvision.utils.save_image(output_steg_1, GAP_SECRET_PATH + 'stego.png')
        torchvision.utils.save_image(rev_secret_1, GAP_SECRET_PATH + 'receive.png')
        # #torchvision.utils.save_image(secret_2, c.TEST_PATH_secret_2 + '%.5d.png' % i)
        # 
        # torchvision.utils.save_image(output_steg_1, c.TEST_PATH_steg_1 + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_1, c.TEST_PATH_secret_rev_1 + '%.5d.png' % i)
        print(i)
        #torchvision.utils.save_image(output_steg_2, c.TEST_PATH_steg_2 + '%.5d.png' % i)
        #torchvision.utils.save_image(rev_secret_2, c.TEST_PATH_secret_rev_2 + '%.5d.png' % i)