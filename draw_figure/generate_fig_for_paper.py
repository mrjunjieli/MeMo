import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import cv2 as cv
import random 
import sys 
import matplotlib.pyplot as plt
sys.path.append("../")
from data_preparation.Visual_perturb import *
import pdb 

# roiSize=112 

# norm_visual = cv.imread('./visual.jpg') # 454 450 3 

# visual_missing = np.zeros_like(norm_visual)
# cv2.imwrite('./visual_missing.jpg', visual_missing, params=None)


# roi =  cv.resize(norm_visual, (450 //10, 450 //10))
# visual_ds = cv.resize(roi, (450 , 450))
# cv2.imwrite('./visual_ds.jpg', visual_ds, params=None)


# blur = torchvision.transforms.GaussianBlur( kernel_size=(13, 13),sigma=(4, 8),)
# visual_gauss_blur = blur(torch.tensor(norm_visual).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0).numpy()
# cv2.imwrite('./visual_gb.jpg', visual_gauss_blur, params=None)

# var = random.uniform(0.02, 0.2)
# visual_zero_mean = random_noise(norm_visual, mode="gaussian", mean=0,var=var, clip=True,)* 255
# visual_zero_mean = np.uint8(visual_zero_mean)
# cv2.imwrite('./visual_zm.jpg', visual_zero_mean, params=None)


# occlude_img, occluder_img, occluder_mask = get_occluders('/home/junjie/AVTSE_momentum/data_preparation/Asset/object_image_sr', 
# '/home/junjie/AVTSE_momentum/data_preparation/Asset/object_mask_x4', state='train')
# alpha_mask = occluder_mask/255.0
# pdb.set_trace()
# alpha_mask = np.expand_dims(occluder_mask, axis=2)/255.0
# alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
# offset_x = 30
# offset_y = 30
# x = 450 / 2
# y = 450 / 2
# lip_concel = overlay_image_alpha(norm_visual, occluder_img, int(x - offset_x), int(y - offset_y), alpha_mask )
# cv2.imwrite('./lip_concel.jpg', lip_concel, params=None)



