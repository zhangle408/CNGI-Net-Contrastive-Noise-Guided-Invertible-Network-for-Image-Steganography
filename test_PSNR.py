'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
from natsort import natsorted
# from skimage.measure import compare_ssim as SSIM, compare_psnr as PSNR
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = '/data/zhangle/third_noise/PARIS_s1_8b_v5_contrastiveloss_v4_1/test_v/c1s2/s/gt/'
    folder_Gen = '/data/zhangle/third_noise/PARIS_s1_8b_v5_contrastiveloss_v4_1/test_v/c1s2/s/c/'
    crop_border = 1
    suffix = '_secret_rev'  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    # base_name1='cover_0'
    # base_name2='container_0'
    base_name1='secret_0'
    base_name2='rev_secret_0'
    im_GT = cv2.imread(os.path.join(folder_GT, base_name1 + '.png')) / 255.
        # print(base_name)
        # print(img_path)
        # print(os.path.join(folder_Gen, base_name + '.png'))
    im_Gen = cv2.imread(os.path.join(folder_Gen, base_name2 + '.png')) / 255.


    if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im_GT_in = bgr2ycbcr(im_GT)
        im_Gen_in = bgr2ycbcr(im_Gen)
    else:
        im_GT_in = im_GT
        im_Gen_in = im_Gen

    # # crop borders
    # if im_GT_in.ndim == 3:
    #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
    #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
    # elif im_GT_in.ndim == 2:
    #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
    #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
    # else:
    #     raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

    # calculate PSNR and SSIM
    print(folder_Gen)
    PSNR = calculate_psnr(im_GT_in, im_Gen_in)
    print('PSNR', PSNR)

    SSIM = calculate_ssim(im_GT_in, im_Gen_in)
    print('SSIM', SSIM)
    ##################################################
    N=1
    psnr_c = np.zeros((3))
    # for i in range(N):
    psnr_c[0] = PSNR(im_GT_in[:, :, 0], im_Gen_in[:, :, 0])
    psnr_c[1] = PSNR(im_GT_in[:, :, 1], im_Gen_in[:, :, 1])
    psnr_c[2] = PSNR(im_GT_in[:, :, 2], im_Gen_in[:, :, 2])
    print("Avg. PSNR C:", psnr_c)

    # SSIM
    ssim_c = np.zeros(N)
    for i in range(N):
        ssim_c[i] = SSIM(im_GT_in[i], im_Gen_in[i], multichannel=True)
    print("Avg. SSIM C:", ssim_c)
    
# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr(origin, pred):
      return PSNR(origin, pred)

def calculate_ssim(img1, img2):
     return  SSIM(img1, img2, multichannel=True)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

#def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
