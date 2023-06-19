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

    if config.model.render_pe:
        loss_pe = compute_pe_loss(config, sample, pred['pe2d_pred'], pred['pe2d_render'])
        losses.update({
            'loss_pe': loss_pe,
            'weight_pe': config.loss_weight_pe
        })

    return losses


def compute_pe_loss(config, sample, pe_pred, pe_render, device):
    '''
    pe_pred and pe_render: in shape [b*t,c,h,w]
    '''
    b,t,c,h,w = sample['images'].shape
    device = pe_pred.device
    feat_size = 16
    masks = sample['fg_probabilities'].to(device)    # [b,t,h,w]
    masks = masks.reshape(b*t,1,h,w)
    masks_down = F.interpolate(masks, [feat_size]*2, mode='nearest')

    pe_target = pe_render.detach() * masks_down.float()
    pe_pred = pe_pred * masks_down.float()
    
    loss_pe = F.mse_loss(pe_pred, pe_target)
    return loss_pe

