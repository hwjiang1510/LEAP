import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.cross_attention import TransformerCrossAttnLayer
from einops import rearrange
import math

class CrossViewEncoder(nn.Module):
    def __init__(self, config, in_dim, in_res) -> None:
        super(CrossViewEncoder, self).__init__()
        self.config = config

        latent_dim = int(4 * in_dim)

        self.num_layers = config.model.encoder_layers
        self.transformers_cross, self.transformers_self = [], []
        for _ in range(self.num_layers):
            cross_attn = TransformerCrossAttnLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim, 
                                                          dropout=0.0, activation='gelu', batch_first=True)
            self_attn = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                                         dropout=0.0, activation='gelu', batch_first=True)
            self.transformers_cross.append(cross_attn)
            self.transformers_self.append(self_attn)
        self.transformers_cross = nn.ModuleList(self.transformers_cross)
        self.transformers_self = nn.ModuleList(self.transformers_self)

        # 2d positional embedding
        embedding_stdev = (1. / math.sqrt(in_dim))
        self.pixel_emb = nn.parameter.Parameter(torch.rand(1, in_dim, in_res, in_res) * embedding_stdev)
        # camera ID embedding (see scene representation transformer)
        self.cam_emb = nn.parameter.Parameter(torch.rand(10, in_dim, 1, 1) * embedding_stdev)
        
        
    def forward(self, x):
        '''
        x in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape

        # add 2D positional embedding (same for each image)
        x = x + self.pixel_emb.unsqueeze(0)                             # [b,t,c,h,w]
        # add camera ID embedding (different between images, same for all pixels in one image)
        cam_emb = self.cam_emb[:t].repeat(1,1,h,w).unsqueeze(0)
        x = x + cam_emb                                                 # [b,t,c,h,w]

        # get canonical view
        x_canonical = x[:, 0]                                           # [b,c,h,w]
        x = x[:, 1:]                                                    # [b,t-1,c,h,w]
        x_canonical = rearrange(x_canonical, 'b c h w -> b (h w) c')
        x = rearrange(x, 'b t c h w -> b (t h w) c')

        # get through transformer encoder
        for (cross_attn, self_attn) in zip(self.transformers_cross, self.transformers_self):   # requires [b,n,c] inputs
            # cross-attention between canonical-other frames
            x = cross_attn(tgt=x, memory=x_canonical)       # [b,(t-1)*h*w,c]
            # concat all frame features
            x = rearrange(x, 'b (t h w) c -> b t c h w', t=t-1, h=h, w=w)
            x_canonical = rearrange(x_canonical, 'b (t h w) c -> b t c h w', t=1, h=h, w=w)
            x = torch.cat([x_canonical, x], dim=1)          # [b,t,c,h,w]
            x = rearrange(x, 'b t c h w -> b (t h w) c')    # [b,n=t*h*w,c]
            # self-attention refinement for all frames
            x = self_attn(x)
            # split the canonical and other frame features
            x = rearrange(x, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
            x_canonical = x[:, 0]
            x = x[:, 1:]
            x_canonical = rearrange(x_canonical, 'b c h w -> b (h w) c')
            x = rearrange(x, 'b t c h w -> b (t h w) c')

        x_canonical = rearrange(x_canonical, 'b (t h w) c -> b t c h w', t=1, h=h, w=w)
        x = rearrange(x, 'b (t h w) c -> b t c h w', t=t-1, h=h, w=w)
        x = torch.cat([x_canonical, x], dim=1)          # [b,t,c,h,w]
        return x
