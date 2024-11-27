#!/usr/bin/env python
import sys
import os
import torch
import torch.nn
import torch.optim
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import tqdm
from model import *
from imp_subnet import *
import torchvision.transforms as T
import config_image_s1 as c
from discriminator import Discriminator
from CHL_builder import CHL
from tensorboardX import SummaryWriter
from datasets_image_s1 import trainloader, testloader
import viz
import modules.module_util as mutil
import modules.Unet_common as common
import warnings
import util
import logging
# from skimage.measure import compare_ssim as SSIM, compare_psnr as PSNR

# from vgg_loss import VGGLoss
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# if torch.cuda.is_available():
# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, ""so you should probably run with --cuda")

def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)

    if mse < 1.0e-10:
        return 100

    if mse > 1.0e15:
        return -100

    return 10 * math.log10(255.0 ** 2 / mse)

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


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.to(device)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = c.lr * ((10**(-0.5)) ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).to(device)

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


bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
cos_loss = nn.CosineSimilarity(dim=1).to(device)
cover_label = 1
stego_label = 0


#####################
# Model initialize: #
#####################
net1 = Model_1()
net_dis = CHL()
net1.cuda()
net_dis.cuda()
init_model(net1)
init_model(net_dis)


net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
net_dis = torch.nn.DataParallel(net_dis, device_ids=device_ids)

para1 = get_parameter_number(net1)
para_dis = get_parameter_number(net_dis)

print(para1)
print(para_dis)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable_dis = (list(filter(lambda p: p.requires_grad, net_dis.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim_dis = torch.optim.Adam(params_trainable_dis, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler_dis = torch.optim.lr_scheduler.StepLR(optim_dis, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix_load + '_1.pt', net1, optim1)
    load(c.MODEL_PATH + c.suffix_load + '_dis.pt', net_dis, optim_dis)


if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_dis.pt', net_dis, optim_dis)


util.setup_logger('train', '/data/CNGI-Net/newlogging', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')
logger_train.info(net1)
logger_train.info(net_dis)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for i_epoch in range(c.epochs):

        net1.train()
        net_dis.train()
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        loss_history_g1 = []
        loss_history_r1 = []
        APD_C = []
        APD_S =[]
        #################
        #     train:    #
        #################
        for i_batch, datalist in enumerate(trainloader):
            # data preparation
            data, labels = datalist
            data = data.to(device)
            cover = data[data.shape[0] // 2:]  # channels = 3
            secret_1 = data[:data.shape[0] // 2]
            cover_dwt = dwt(cover)  # channels = 12
            secret_dwt_1 = dwt(secret_1)
            input_guass = gauss_noise(cover_dwt.shape)
            #print(cover_dwt.shape)
            #print(secret_dwt_1.shape)
            input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24
            input_hide = torch.cat((input_dwt_1, input_guass), 1)

            #################
            #    forward1:   #
            #################
            # with torch.enable_grad():
            output_dwt_1 = net1(input_hide)  # channels = 24
            output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 8 * c.channels_in)  # channels = 12

            # get steg1
            output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3



            output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
            output_rev_dwt_1 = torch.cat((output_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

            rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

            rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
            rev_secret_1 = iwt(rev_secret_dwt).to(device)
            APD_temp_S = (rev_secret_1 - secret_1).abs().mean() * 255
            APD_temp_C = (output_steg_1 - cover).abs().mean() * 255

            '''=========================================================='''
            p1, p2, z1, z2 = net_dis(x1=cover, x2=output_steg_1)
            contrastive_loss = -(cos_loss(p1, z2).mean() + cos_loss(p2, z1).mean()) * 0.5

            '''=========================================================='''





            #################
            #     loss:     #
            #################
            g_loss_1 = guide_loss(output_steg_1.cuda(), cover.cuda())
            r_loss_1 = reconstruction_loss(rev_secret_1, secret_1)

            total_loss = c.lamda_reconstruction_1 * r_loss_1 + c.lamda_guide_1 * g_loss_1 + c.contrastive_loss_weight * contrastive_loss#c.adversarial_loss * g_loss_adv#+ c.lamda_low_frequency_1 * l_loss_1
            total_loss.backward()
            if i_batch % 1000 == 0:
                print('---------')
                print('i_batch', i_batch, 'APD_temp_C', APD_temp_C, 'APD_temp_S', APD_temp_S, )
                print('r_loss', r_loss_1, 'g_loss', g_loss_1)
                print('---------')


            if c.optim_step_1:
                optim1.step()

            optim1.zero_grad()
            optim_dis.zero_grad()
            loss_history.append([total_loss.item(), 0.])
            loss_history_g1.append(g_loss_1.item())
            loss_history_r1.append(r_loss_1.item())

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                print('############val##########')
                psnr_s1 = []
                APD_S1 =[]
                psnr_c1 = []
                APD_C1=[]
                net1.eval()
                net_dis.eval()
                for i_batch, xlist in enumerate(testloader):
                    x,labels = xlist
                    x = x.to(device)
                    cover = x[x.shape[0] // 2:]  # channels = 3
                    secret_1 = x[:x.shape[0] // 2]

                    cover_dwt = dwt(cover)  # channels = 12
                    secret_dwt_1 = dwt(secret_1)
                    # secret_dwt_2 = dwt(secret_2)
                    input_guass = gauss_noise(cover_dwt.shape)

                    input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24
                    input_hide = torch.cat((input_dwt_1, input_guass), 1)

                    #################
                    #    forward1:   #
                    #################
                    output_dwt_1 = net1(input_hide)  # channels = 24
                    output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 8 * c.channels_in)  # channels = 12

                    # get steg1
                    output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3



                    output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
                    output_rev_dwt_1 = torch.cat((output_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

                    rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

                    rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                    rev_secret_1 = iwt(rev_secret_dwt).to(device)

                    '''=========================================================='''
                    p1, p2, z1, z2 = net_dis(x1=cover, x2=output_steg_1)
                    loss = -(cos_loss(p1, z2).mean() + cos_loss(p2, z1).mean()) * 0.5

                    '''=========================================================='''

                    secret_rev1_255 = rev_secret_1.cpu().numpy().squeeze() * 255
                    secret_1_255 = secret_1.cpu().numpy().squeeze() * 255

                    cover_255 = cover.cpu().numpy().squeeze() * 255
                    steg_1_255 = output_steg_1.cpu().numpy().squeeze() * 255

                    psnr_temp1 = computePSNR(secret_rev1_255, secret_1_255)
                    psnr_s1.append(psnr_temp1)

                    psnr_temp_c1 = computePSNR(cover_255, steg_1_255)
                    psnr_c1.append(psnr_temp_c1)
                writer.add_scalars("PSNR", {"S1 average psnr": np.mean(psnr_s1)}, i_epoch)
                writer.add_scalars("PSNR", {"C1 average psnr": np.mean(psnr_c1)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'PSNR_S: {np.mean(psnr_s1):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c1):.4f} | '
                )

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim1.param_groups[0]['lr'])

        epoch_losses_g1 = np.mean(np.array(loss_history_g1))
        epoch_losses_r1 = np.mean(np.array(loss_history_r1))
        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
        writer.add_scalars("Train", {"g1_Loss": epoch_losses_g1}, i_epoch)
        writer.add_scalars("Train", {"r1_Loss": epoch_losses_r1}, i_epoch)

        logger_train.info(f"Learning rate: {optim1.param_groups[0]['lr']}")
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
        )

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim1.state_dict(),
                        'net': net1.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_1' % i_epoch + '.pt')
            torch.save({'opt': optim_dis.state_dict(),
                        'net': net_dis.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_dis' % i_epoch + '.pt')

        weight_scheduler1.step()
        weight_scheduler_dis.step()


    torch.save({'opt': optim1.state_dict(),
                'net': net1.state_dict()}, c.MODEL_PATH + 'model_1' + '.pt')
    torch.save({'opt': optim_dis.state_dict(),
                'net': net_dis.state_dict()}, c.MODEL_PATH + 'model_dis' + '.pt')

    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH + 'model_ABORT_1' + '.pt')
        torch.save({'opt': optim_dis.state_dict(),
                    'net': net_dis.state_dict()}, c.MODEL_PATH + 'model_ABORT_dis' + '.pt')

    raise

finally:
    viz.signal_stop()
