import torch.nn as nn
import torch
import torch.nn.functional as F


def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))
        down_rate = 14
        if type == 'vitb14':
            backbone_dim = 768
        elif type == 'vits14':
            backbone_dim = 384
    return backbone, down_rate, backbone_dim