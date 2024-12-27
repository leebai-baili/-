from typing import List, Dict, Tuple

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

import torch
import torchvision

import torch.nn.functional as F



def torch_PSNR(image_true, image_test, data_range=1.):
    mse = torch.mean((image_true - image_test) ** 2)
    return 10 * torch.log10(data_range**2 / mse)


def torch_SSIM(image_true_, image_test_, data_range=1., window_size=11, sigma=1.5):
    temp = 0
    for i in range(3):
        image_true = image_true_[:, i, ...].unsqueeze(1)
        image_test = image_test_[:, i, ...].unsqueeze(1)
        # Calculate constants for SSIM
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # Create a Gaussian window
        window = torch.exp(-(torch.linspace(-(window_size // 2), window_size // 2, window_size) ** 2) / (2.0 * sigma ** 2))
        window = window / window.sum()
        #window = window.cuda()

        # Compute local
        mu_true = F.conv2d(image_true, window.view(1, 1, window_size, 1), padding=0, groups=1)
        mu_test = F.conv2d(image_test, window.view(1, 1, window_size, 1), padding=0, groups=1)

        # Compute local variances
        mu_true_sq = mu_true * mu_true
        mu_test_sq = mu_test * mu_test
        mu_true_test = mu_true * mu_test

        # Compute local variances
        sigma_true_sq = F.conv2d(image_true * image_true, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_true_sq
        sigma_test_sq = F.conv2d(image_test * image_test, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_test_sq
        sigma_true_test = F.conv2d(image_true * image_test, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_true_test

        # Compute SSIM map
        ssim_map = ((2 * mu_true_test + C1) * (2 * sigma_true_test + C2)) / ((mu_true_sq + mu_test_sq + C1) * (sigma_true_sq + sigma_test_sq + C2))

        temp += ssim_map.mean()

    # Return the mean SSIM value of the entire image
    return temp/3

import cv2
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from albumentations import (Compose,
    HorizontalFlip,  Resize,
    Normalize)

def get_train_transforms():
    # p:使用此转换的概率，默认值为 0.5
    return Compose([
        # HorizontalFlip(p=0.5),  # 水平翻转
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0), # 归一化，将像素值除以255，减去每个通道的平均值并除以每个通道的std
    ], p=1.)


def get_valid_transforms():
    return Compose([
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], p=1.)


a = torch.rand(1, 3, 256, 256)
b = torch.rand(1, 3, 256, 256)

c = torch.rand(1, 3, 256, 256)*255
d = torch.rand(1, 3, 256, 256)*255
#torch_PSNR(d, c)
print(torch_PSNR(a, b))
print(torch_PSNR(b, b))
print(torch_SSIM(a, b))
print(torch_SSIM(b, b))


image_path = '/home/gem/GuanH/Program/Data/dault_image/0112_0080/00000.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_dict = {'image': image}

transforms = get_valid_transforms()
transformed = transforms(**input_dict)
normalized_image = transformed['image']
print(normalized_image.shape) 
print("Normalized min:", normalized_image.min())
print("Normalized max:", normalized_image.max())


