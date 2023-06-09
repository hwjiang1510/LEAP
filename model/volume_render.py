import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Volumes
from pytorch3d.renderer import VolumeRenderer, NDCGridRaysampler, EmissionAbsorptionRaymarcher
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.renderer.cameras import PerspectiveCameras
from utils.train_utils import init_weights_conv
import math
from utils.vis_utils import unnormalize, normalize


class VolRender(nn.Module):
    def __init__(self, config):
        super(VolRender, self).__init__()
        self.config = config

        # render image resolution setting
        self.img_input_res = config.dataset.img_size
        self.img_render_res = config.dataset.img_size_render
        self.render_down_rate = self.img_input_res // self.img_render_res

        # neural volume physical world settings
        self.volume_physical_size = config.render.volume_size

        # build renderer
        self.raySampler = NDCGridRaysampler(image_width=self.img_render_res,
                                            image_height=self.img_render_res,
                                            n_pts_per_ray=config.render.n_pts_per_ray,
                                            min_depth=config.render.min_depth,
                                            max_depth=config.render.max_depth)
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)

        # build feature->rgb upsample module
        self.k_size = config.render.k_size
        self.pad_size = self.k_size // 2
        self.render_feat_dim = config.model.render_feat_dim
        self.upsample_conv = []
        for _ in range(int(math.log2(self.render_down_rate))):
            self.upsample_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.render_feat_dim, self.render_feat_dim, kernel_size=self.k_size+1, stride=2, padding=self.pad_size),
                    nn.BatchNorm2d(self.render_feat_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.upsample_conv.append(
            nn.Sequential(
                nn.Conv2d(self.render_feat_dim, 8, kernel_size=self.k_size, stride=1, padding=self.pad_size),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(8, 3, kernel_size=self.k_size, stride=1, padding=self.pad_size),
            )   
        )
        self.upsample_conv = nn.Sequential(*self.upsample_conv)

        # mean_weight = -2.0 if config.train.normalize_img else 0.0
        # init_weights_conv(self.upsample_conv[-1], mean_weight=mean_weight, mean_bias=0.0)


    def forward(self, camera_params, features, densities, render_depth=False):
        '''
        camera_params: pytorch3d perspective camera, parameters in batch size B
        features: [B,C,D,H,W]
        densities: [B,1,D,H,W]
        '''
        B,C,D,H,W = features.shape
        device = features.device

        # parse camera parameters considering render downsample rate
        camera_params['K'] /= self.render_down_rate
        camera_params['K'][:,-1,-1] = 1.0
        cameras = cameras_from_opencv_projection(R=camera_params['R'],
                                                 tvec=camera_params['T'], 
                                                 camera_matrix=camera_params['K'],
                                                 image_size=torch.tensor([self.img_render_res]*2).unsqueeze(0).repeat(B,1)).to(device)
        
        # parse neural volume physical world settings
        single_voxel_size = self.volume_physical_size / D
        volume = Volumes(densities=densities,
                         features=features,
                         voxel_size=single_voxel_size)
        
        # perform neural rendering
        rendered = self.renderer(cameras=cameras, volumes=volume, render_depth=render_depth)[0]  # [B,H,W,C+1]

        # split into rgb, mask and depth, and get to original input resolution
        if not render_depth:
            rendered_imgs, rendered_mask = rendered.split([C,1], dim=-1)
        else:
            rendered_imgs, rendered_mask, rendered_depth = rendered.split([C,1,1], dim=-1)
            rendered_depth = rendered_depth.permute(0,3,1,2).contiguous()
            rendered_depth = F.upsample(rendered_depth, size=[self.img_input_res]*2, mode='bilinear')
        rendered_imgs = rendered_imgs.permute(0,3,1,2).contiguous()
        rendered_mask = rendered_mask.permute(0,3,1,2).contiguous()
        rendered_imgs = self.upsample_conv(rendered_imgs)
        if self.config.train.normalize_img:
            rendered_imgs = unnormalize(rendered_imgs)
        rendered_imgs = F.relu(rendered_imgs)
        if self.config.train.normalize_img:
            rendered_imgs = normalize(rendered_imgs)
        rendered_mask = F.upsample(rendered_mask, size=[self.img_input_res]*2, mode='bilinear')

        results = {
            'rgb': rendered_imgs,       # [B=b*t,3,h,w]
            'mask': rendered_mask,      # [B,1,h,w] 
        }
        if render_depth:
            results['depth']: render_depth  # [B,1,h,w]

        return results


        
        