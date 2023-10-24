#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from raft.utils import flow_viz
from math import exp
import argparse
import numpy as np
from raft.raft import RAFT

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def rigid(xyz, xyz_prev, R, R_prev, k=10):
    loss = 0
    for i in range(xyz.shape[0]):
        loss += _rigid(xyz[i], xyz, R, xyz_prev[i], xyz_prev, R_prev, k) ##! 改成batch操作
    return loss 

def _rigid(xyz_i, xyz, R, xyz_prev_i, xyz_prev, R_prev, k=10):
    ## 对于所有的gaussian点，随机选择k个点
    """
    p_i: 1, 3
    xyz: N, 3
    R_prev: 3, 3
    R: 3, 3
    """    
    j_indices = torch.randint(0, xyz.shape[0], (k,)) #k,
    xyz_j = xyz[j_indices] # k, 3
    xyz_prev_j = xyz_prev[j_indices]

    diff = (xyz_prev_j - xyz_prev_i) - (R_prev @ R.T @ (xyz_j - xyz_i).T).T # k, 3
    return (diff ** 2).mean()


def mask_loss(logits, target):
    criterion = torch.nn.BCELoss()
    loss = criterion(logits, target)
    return loss

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return (1 - ((2. * intersection + smooth) / 
                 (pred.sum() + target.sum() + smooth)))

def iou_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (1 - (intersection + smooth) / (union + smooth))



def gradient_magnitude(x):
    # 使用Sobel滤波器计算梯度
    sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(x.device)
    sobel_x = sobel_x.repeat(3, 1, 1, 1)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=3)  # 使用groups参数以逐通道方式应用滤波器
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=3)
    
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    avg_magnitude = magnitude.mean(dim=1, keepdim=True)  # 沿通道方向平均
    
    return avg_magnitude

def weighted_mse_loss(pred, target, alpha=2.0):
    # pred 和 target 都是 Bx3xHxW 的图像张量
    grad_mag = gradient_magnitude(target)
    weights = torch.exp(alpha * grad_mag / grad_mag.max())  # 对梯度幅度进行归一化和缩放
    
    mse_loss = (pred - target) ** 2
    weighted_loss = mse_loss * weights
    
    return weighted_loss.mean()



class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)
    
class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss
    
class Optical_loss(torch.nn.Module):
    def __init__(self) -> None:
        super(Optical_loss, self).__init__()
        self.args = self.get_args_from_dict({
            "model": "raft/trained_model/raft-things.pth",
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.cuda()
        # self.model.eval()

    def forward(self, image1, image2, gt_mask_vis, save_path, save_fig=False):
        image1, image2 = image1.unsqueeze(0)*255, image2.unsqueeze(0)*255
        flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        flow_up = flow_up * gt_mask_vis
        if save_fig:
            self.viz(image1, flow_up, str(save_path))
        return torch.square(flow_up).mean()
    
    @staticmethod
    def viz(img, flo, save_path):
        img = img[0].detach().permute(1,2,0).cpu().numpy()
        flo = flo[0].detach().permute(1,2,0).cpu().numpy()
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        cv2.imwrite(save_path, img_flo[:, :, [2,1,0]].astype(np.uint8))
    
    @staticmethod
    def get_args_from_dict(data_dict):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="raft/trained_model/raft-things.pth", help="restore checkpoint")
        parser.add_argument('--path', default="raft/demo-frames", help="dataset for evaluation")
        parser.add_argument('--small', default=False, action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

        # Construct a mock list of arguments from the dictionary
        mock_argv = []
        for key, value in data_dict.items():
            if isinstance(value, bool):
                if value:
                    mock_argv.append("--"+key)
            else:
                mock_argv.extend(["--"+key, str(value)])
        
        # Parse the mock list of arguments
        return parser.parse_args(mock_argv)