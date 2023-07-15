import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.cross_attention import TransformerCrossAttnLayer
from model.base_module.TransformerDecoder import TransformerDecoderLayerPermute
from model.base_module.flash_attention.transformer import FlashTxDecoderLayer, FlashCrossAttnLayer
from einops import rearrange
import math


class lifting(nn.Module):
    def __init__(self, config, in_dim) -> None:
        super(lifting, self).__init__()
        self.config = config

        embedding_stdev = (1. / math.sqrt(in_dim))
        self.latent_res = config.model.latent_res
        self.latent_emb = nn.parameter.Parameter(
                            (torch.rand(self.latent_res, self.latent_res, self.latent_res, in_dim) * embedding_stdev))

        if config.model.use_pe_lifting:
            self.lift_init = lifting_make_init_layer(config, in_dim)
        
        self.transformer = lifting_make_transformer_layers(config, in_dim)
        #self.conv3ds = lifting_make_conv3d_layers(config, in_dim)

        self.latent_refine = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 256, 4, stride=2, padding=1),
            #nn.Conv3d(in_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
        )
    

    def forward(self, x, pe2d):
        '''
        x: 2D features in shape [b,t,c,h,w]
        pe2d: 2D p.e. in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape
        device = x.device
        x = rearrange(x, 'b t c h w -> b (t h w) c')
        if torch.is_tensor(pe2d):
            pe2d = rearrange(pe2d, 'b t c h w -> b (t h w) c')
        
        latent = rearrange(self.latent_emb, 'd h w c -> (d h w) c').unsqueeze(0).repeat(b,1,1).to(device)  # [b,N=d*h*w,c]
        #print('pe3d', latent.mean().item(), latent.var().item(), latent.min().item(), latent.max().item())
        
        if self.config.model.use_pe_lifting and (not self.config.model.use_flash_attn):
            #latent = self.lift_init(tgt=latent, memory=pe2d, value=x)       # tgt is query, memory is key
            latent = self.lift_init(latent, pe2d)
        
        if self.config.model.render_feat_raw and self.training:
            latent_raw = latent.clone()
            latent_raw = rearrange(latent_raw, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        else:
            latent_raw = None
        
        if torch.is_tensor(pe2d):
            x = x + pe2d
        
        # for (block_transformer, block_conv3d) in zip(self.transformer, self.conv3ds):
        #     latent = block_transformer(latent, x)
        #     latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        #     latent = block_conv3d(latent)
        #     latent = rearrange(latent, 'b c d h w -> b (d h w) c')
        for block in self.transformer:
            latent = block(latent, x)

        latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        
        # if self.config.model.render_feat_raw and self.training:
        #     latent_raw = latent.clone()
        # else:
        #     latent_raw = None

        latent = self.latent_refine(latent)
        
        return latent_raw, latent



def lifting_make_init_layer(config, in_dim):
    mlp_ratio = 4.0
    norm_first = config.model.norm_first

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        if not config.model.lifting_TXdecoder_permute:
            layer = torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                            dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
        else:
            layer = TransformerDecoderLayerPermute(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                            dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
    else:
        layer = FlashTxDecoderLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
    return layer


def lifting_make_transformer_layers(config, in_dim):
    transformer = []
    num_layers = config.model.lifting_layers
    mlp_ratio = 4.0
    norm_first = config.model.norm_first

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        transformer = [torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim, 
                                                        dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
                       for _ in range(num_layers)]
    else:
        transformer = [FlashTxDecoderLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
                       for _ in range(num_layers)]
    transformer = nn.ModuleList(transformer)
    return transformer


def lifting_make_conv3d_layers(config, in_dim):
    num_layers = config.model.lifting_layers
    if config.model.lifting_use_conv3d:
        conv3ds = [nn.Sequential(nn.Conv3d(in_dim, in_dim, 3, padding=1),
                             nn.BatchNorm3d(in_dim),
                             nn.LeakyReLU(inplace=True),) for _ in range(num_layers)]
    else:
        conv3ds = [nn.Identity() for _ in range(num_layers)]
    conv3ds = nn.ModuleList(conv3ds)
    return conv3ds