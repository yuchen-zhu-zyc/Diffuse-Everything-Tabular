
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multimodal_diffusion_loss(score_num, score_cat_list, noise_num, cat_clean):
    B = score_num.shape[0]
    continuous_loss = 1/B * torch.sum((score_num - noise_num)**2)
    discrete_loss = 0
    for i in range(cat_clean.shape[1]):
        discrete_loss += F.cross_entropy(score_cat_list[i], cat_clean[:, i], reduction = "none")
    
    return continuous_loss, discrete_loss

def multimodal_diffusion_loss_imp(score_num, score_cat_list, noise_num, cat_clean):
    B = score_num.shape[0]
    continuous_loss = 1/B * torch.sum((score_num - noise_num)**2)
    discrete_loss = 0
    discrete_loss_imp = 0
    for i in range(cat_clean.shape[1]):
        if i == 0:
            discrete_loss_imp += F.cross_entropy(score_cat_list[i], cat_clean[:, i], reduction = "none")
        else:
            discrete_loss += F.cross_entropy(score_cat_list[i], cat_clean[:, i], reduction = "none")
        
    
    return continuous_loss, discrete_loss, discrete_loss_imp
