import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math

class lifting(nn.Module):
    def __init__(self, config, in_dim) -> None:
        super(lifting, self).__init__()
        self.config = config

        embedding_stdev = (1. / math.sqrt(in_dim))
        self.latent_res = config.model.latent_res
        self.latent_emb = nn.parameter.Parameter(torch.rand(self.latent_res, self.latent_res, self.latent_res, in_dim) * embedding_stdev)

        self.transformer = []
        for _ in range(config.model.lifting_layers):
            self.transformer.append(
                    torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=int(4*in_dim), 
                                                    dropout=0.0, activation='gelu', batch_first=True)
            )
        self.transformer = nn.ModuleList(self.transformer)

        self.latent_refine = nn.Sequential(
            nn.Conv3d(in_dim, 256, 3, padding=1),
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

        latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        latent = self.latent_refine(latent)

        return latent

