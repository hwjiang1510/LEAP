import os
import pprint
import random
import numpy as np
import torch
import warnings
import torch.nn as nn

from dataset.kubric import Kubric


def get_optimizer(config, model):
    # get pe parameters
    pe_list = ['neck.pe_canonical', 'lifting.latent_emb']
    pe_params = []
    for it in pe_list:
        pe_params += list(filter(lambda kv: it in kv[0], model.named_parameters()))
    
    # get backbone parameters
    backbone_param = list(filter(lambda kv: 'backbone' in kv[0], model.named_parameters()))

    # other parameters
    non_other_param_names = list(map(lambda x: x[0], pe_params)) + list(map(lambda x: x[0], backbone_param))
    other_param = [(name, param) for name, param in model.named_parameters() if name not in non_other_param_names]

    # remove param names
    pe_params = list(map(lambda x: x[1], pe_params))
    backbone_param = list(map(lambda x: x[1], backbone_param))
    other_param = list(map(lambda x: x[1], other_param))

    # get optimizer
    if config.model.backbone_fix:
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings},
                                       {'params': other_param, 'lr': config.train.lr}],
                                      lr=config.train.lr,
                                      weight_decay=config.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings},
                                       {'params': backbone_param, 'lr': config.train.lr},
                                       {'params': other_param, 'lr': config.train.lr}],
                                       lr=config.train.lr,
                                       weight_decay=config.train.weight_decay)
    return optimizer


def get_dataset(config, split='train'):
    name = config.dataset.name
    if name == 'kubric':
        data = Kubric(config, split=split)
    else:
        raise NotImplementedError('not implemented dataset')
    return data


def resume_training(model, optimizer, schedular, scaler, output_dir, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    output_dir = os.path.join(output_dir, cpt_name)
    if os.path.isfile(output_dir):
        print("=> loading checkpoint {}".format(output_dir))
        if device is not None:
            checkpoint = torch.load(output_dir, map_location=device)
        else:
            checkpoint = torch.load(output_dir, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load schedular
        schedular.load_state_dict(checkpoint['schedular'])

        # load scaler
        scaler.load_state_dict(checkpoint['scaler'])

        # load epoch
        start_epoch = checkpoint['epoch']

        # load data
        best_psnr = checkpoint['best_psnr'] if 'best_psnr' in checkpoint.keys() else 0.0

        return model, optimizer, schedular, scaler, start_epoch, best_psnr
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(output_dir))
    
def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def init_weights_conv(m, mean_weight, mean_bias):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=mean_weight, std=1e-4)
        nn.init.normal_(m.bias, mean=mean_bias, std=1e-4)