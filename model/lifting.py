import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.flash_attention.transformer import FlashTxDecoderLayer
from einops import rearrange
import math


class lifting(nn.Module):
    def __init__(self, config, in_dim) -> None:
        super(lifting, self).__init__()
        self.config = config

        embedding_stdev = (1. / math.sqrt(in_dim))
        dtype = torch.float16 if self.config.model.use_flash_attn else torch.float32
        self.latent_res = config.model.latent_res
        self.latent_emb = nn.parameter.Parameter(
                            (torch.rand(self.latent_res, self.latent_res, self.latent_res, in_dim) * embedding_stdev).to(dtype))

        self.transformer = lifting_make_transformer_layers(config, in_dim)

        self.latent_refine = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 256, 4, stride=2, padding=1),
            #nn.Conv3d(in_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
        )
    

    def forward(self, x):
        '''
        x: 2D features in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape
        device = x.device
        x = rearrange(x, 'b t c h w -> b (t h w) c')
        
        latent = rearrange(self.latent_emb, 'd h w c -> (d h w) c').unsqueeze(0).repeat(b,1,1).to(device)  # [b,N=d*h*w,c]
        
        for block in self.transformer:
            latent = block(latent, x)

        if self.config.model.use_flash_attn:
            latent = latent.float()

        latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        latent = self.latent_refine(latent)

        return latent


def lifting_make_transformer_layers(config, in_dim):
    transformer = []
    num_layers = config.model.lifting_layers
    mlp_ratio = 4.0

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        transformer = [torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim, 
                                                            dropout=0.0, activation='gelu', batch_first=True)
                       for _ in range(num_layers)]
    else:
        transformer = [FlashTxDecoderLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=False)
                       for _ in range(num_layers)]
    transformer = nn.ModuleList(transformer)
    return transformer
