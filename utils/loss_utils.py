import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_losses(config, iter_num, pred, sample, perceptual_loss=None):
    losses = {}
    rendered_imgs, rendered_masks = pred['rgb'], pred['mask']   # [b*t,c,h,w]
    device = rendered_imgs.device

    imgs = sample['images'].to(device)                          # [b,t,c,h,w]
    masks = sample['fg_probabilities'].to(device)
    b,t,c,h,w = imgs.shape
    target_imgs = imgs.view(b*t,c,h,w)
    target_masks = masks.view(b*t,1,h,w)

    loss_rgb = F.mse_loss(rendered_imgs, target_imgs)
    loss_mask = F.mse_loss(rendered_masks, target_masks)
    if perceptual_loss is not None and iter_num > config.loss.iter_perceptual:
        loss_perceptual = perceptual_loss(rendered_imgs, target_imgs, already_normalized=config.train.normalize_img)
        losses['loss_perceptual'] = loss_perceptual
    else:
        loss_perceptual = 0.0

    losses.update({
        'loss_render_rgb': loss_rgb,
        'loss_render_mask': loss_mask,
        'weight_render_rgb': config.loss.weight_render_rgb,
        'weight_render_mask': config.loss.weight_render_mask,
        'weight_perceptual': config.loss.weight_perceptual
    })
    return losses 
