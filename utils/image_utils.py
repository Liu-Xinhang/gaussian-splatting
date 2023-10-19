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
from PIL import Image
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def PutTheSecondImageOnTheFirstImageWithOpacity(img1, img2, opacity=128):
    img2 = (np.transpose(img2, (1, 2, 0)) * 255).astype(np.uint8)
    img1 = (np.transpose(img1, (1, 2, 0)) * 255).astype(np.uint8)
    img2 = Image.fromarray(img2)
    img1 = Image.fromarray(img1)
    img2 = img2.resize(img1.size)

    # 设置透明度并叠加
    img2.putalpha(opacity)  # 设置透明度，范围是0(完全透明)到255(完全不透明)
    img1.paste(img2, (0, 0), img2)
    return img1
