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
from CHL_builder import CHL
from model import *
# from imp_subnet import *
import config_image_s1 as c
from os.path import join
import datasets_image_s1
import modules.module_util as mutil
import modules.Unet_common as common
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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
net_dis = CHL()
net1.cuda()
net_dis.cuda()
init_model(net1)
init_model(net_dis)
net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
net2 = torch.nn.DataParallel(net_dis, device_ids=c.device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
dwt = common.DWT()
iwt = common.IWT()

if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2)



with torch.no_grad():
    net1.eval()
    net_dis.eval()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    psnr_C = AverageMeter()
    psnr_S = AverageMeter()
    ssim_C = AverageMeter()
    ssim_S = AverageMeter()
    lpips_C = AverageMeter()
    lpips_S = AverageMeter()
    #net2.eval()
    #net3.eval()
    import time
    start = time.time()
    # for i, xx in enumerate(datasets.testloader):
    #     x, lable = xx
    #     x = x.to(device)
    #
    #     cover = x[x.shape[0] // 2:]  # channels = 3
    #     secret_1 = x[:x.shape[0] // 2]
    for i, ((secret_1, secret_target), (cover, cover_target)) in enumerate(datasets_image_s1.test_v_loader, 0):
        print('===================begin===========================')
        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        #print(x.shape)
        #print(cover.shape)
        #print(secret_1.shape)
        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        input_guass = gauss_noise(cover_dwt.shape).to(device)
        #secret_dwt_2 = dwt(secret_2)
        #print(cover_dwt.shape)
        #print(secret_dwt_1.shape)
        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1).to(device) # channels = 24
        input_hide = torch.cat((input_dwt_1, input_guass), 1).to(device)


        #################
        #    forward1:   #
        #################
        output_dwt_1 = net1(input_hide)  # channels = 24
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 8 * c.channels_in)  # channels = 12

        # get steg1
        output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3

        '''torchvision.utils.save_image(cover, c.TEST_PATH_cover + '%.5d.png' % i)
                        torchvision.utils.save_image(secret_1, c.TEST_PATH_secret_1 + '%.5d.png' % i)
                        #torchvision.utils.save_image(secret_2, c.TEST_PATH_secret_2 + '%.5d.png' % i)

                        torchvision.utils.save_image(output_steg_1, c.TEST_PATH_steg_1 + '%.5d.png' % i)
                        torchvision.utils.save_image(rev_secret_1, c.TEST_PATH_secret_rev_1 + '%.5d.png' % i)
                        print(i)'''
        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
        output_rev_dwt_1 = torch.cat((output_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

        rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt).to(device)
        rev_secret_img = rev_secret_1
        cover_imgv = cover.to(device)
        container_img = output_steg_1.to(device)
        secret_imgv_nh = secret_1.to(device)
        print('******************************************')


        #C1S1
        dir = '/data/CNGI-Net/num_1/'
        cover_gap = container_img - cover_imgv
        cover_gap_10 = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)

        for i_cover in range(c.batch_size):
            cover_i = cover_imgv[i_cover, :, :, :]
            container_i = container_img[i_cover, :, :, :]
            rev_secret_img_i = rev_secret_img[i_cover, :, :, :]
            cover_gap_10_i = cover_gap_10[i_cover, :, :, :]

        secret_gap = rev_secret_img - secret_imgv_nh
        secret_gap_10 = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)

        for i_secret in range(c.batch_size):
            secret_i = secret_imgv_nh[i_secret, :, :, :]
            rev_secret_i = rev_secret_img[i_secret, :, :, :]
            rev_secret_img_i = rev_secret_img[i_secret, :, :, :]
            secret_gap_10_i = secret_gap_10[i_secret, :, :, :]


        print('done!')




        rev_secret_img = rev_secret_1
        cover_imgv = cover
        container_img = output_steg_1
        secret_imgv_nh = secret_1
        diffH = (container_img - cover_imgv).abs().mean() * 255
        diffR = (rev_secret_img - secret_imgv_nh).abs().mean() * 255
        N, _, _, _ = rev_secret_img.shape

        cover_img_numpy = cover_imgv.clone().cpu().detach().numpy()
        container_img_numpy = container_img.clone().cpu().detach().numpy()

        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)

        rev_secret_numpy = rev_secret_img.cpu().detach().numpy()
        secret_img_numpy = secret_imgv_nh.cpu().detach().numpy()

        rev_secret_numpy = rev_secret_numpy.transpose(0, 2, 3, 1)
        secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)

        # PSNR
        print("Cover Agnostic")

        # print("Secret APD C:", diffH.item())

        psnr_c = np.zeros((N, 3))
        for i in range(N):
            psnr_c[i, 0] = PSNR(cover_img_numpy[i, :, :, 0], container_img_numpy[i, :, :, 0])
            psnr_c[i, 1] = PSNR(cover_img_numpy[i, :, :, 1], container_img_numpy[i, :, :, 1])
            psnr_c[i, 2] = PSNR(cover_img_numpy[i, :, :, 2], container_img_numpy[i, :, :, 2])
        print("Avg. PSNR C:", psnr_c.mean().item())

        # SSIM
        ssim_c = np.zeros(N)
        for i in range(N):
            ssim_c[i] = SSIM(cover_img_numpy[i], container_img_numpy[i], multichannel=True)
        print("Avg. SSIM C:", ssim_c.mean().item())

        psnr_s = np.zeros(N)
        for i in range(N):
            psnr_s[i] = PSNR(secret_img_numpy[i], rev_secret_numpy[i])
        print("Avg. PSNR S:", psnr_s.mean().item())

        # SSIM
        ssim_s = np.zeros(N)
        for i in range(N):
            ssim_s[i] = SSIM(secret_img_numpy[i], rev_secret_numpy[i], multichannel=True)
        print("Avg. SSIM S:", ssim_s.mean().item())

        psnr_S.update(psnr_s.mean().item(), 44 * 1 * 1)  # R loss
        ssim_S.update(ssim_s.mean().item(), 44 * 1 * 1)
        Rdiff.update(diffR.item(), 44 * 1 * 1)
        psnr_C.update(psnr_c.mean().item(), 44 * 1 * 1)  # R loss
        ssim_C.update(ssim_c.mean().item(), 44 * 1 * 1)
        Hdiff.update(diffH.item(), 44 * 1 * 1)
        if i == 200 - 1:
            break
    print('Hdiff.avg, Rdiff.avg', Hdiff.avg, Rdiff.avg)
    print('Hdiff.avg', Hdiff.avg, 'psnr_c.avg', psnr_C.avg, 'ssim_c.avg', ssim_C.avg)
    print('Rdiff.avg', Rdiff.avg, 'psnr_s.avg', psnr_S.avg, 'ssim_s.avg', ssim_S.avg)

