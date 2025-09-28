import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import scipy.stats as stats
from einops import rearrange
from typing import Callable, Optional

import peft
from timm.models import create_model
from timm.layers import trunc_normal_, _assert
if __name__ == '__main__':
    import sys
    sys.path.append('.')
from modelling.modules.timm_vit.to_pixel import ToPixel
from modelling.modules.timm_vit.vision_transformer import Attention
from modelling.modules.timm_vit.vision_transformer import MoVQNorm, MoVQBlockv2
from modelling.modules.timm_vit.rope_utils import compute_mixed_cis, init_random_2d_freqs, init_t_xy
from modelling.modules.timm_vit.rope_utils import compute_axial_cis_3d

class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: int = 16,
        t_patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (t_patch_size, patch_size, patch_size)
        self.img_size = (num_frames, img_size, img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, F, H, W = x.shape
        _assert(
            F % self.patch_size[0] == 0,
            f"Input n_frames ({F}) should be divisible by patch size ({self.patch_size[0]})."
        )
        _assert(
            H % self.patch_size[1] == 0,
            f"Input height ({H}) should be divisible by patch size ({self.patch_size[1]})."
        )
        _assert(
            W % self.patch_size[2] == 0,
            f"Input width ({W}) should be divisible by patch size ({self.patch_size[2]})."
        )
        x = self.proj(x)
        x = rearrange(x, 'b c f h w -> b (f h w) c') # flatten
        x = self.norm(x)
        return x

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class TimmViTEncoder(nn.Module):
    def __init__(self, 
        in_channels=3, num_latent_tokens=32,
        model_name='vit_small_patch14_dinov2.lvd142m',
        model_kwargs={'img_size': 224, 'num_frames': 1, 'patch_size': 14, 't_patch_size': 1, 'drop_path_rate': 0.0,},
        pretrained=True, tuning_method='full', tuning_kwargs={'r': 8},
        rope_theta=100.0, rope_theta_t=100.0, rope_mixed=False, use_rope=False, use_ape=True,
        rope_dim=None, rope_heads=None, rope_layers=None,
        token_drop=0.4, token_drop_max=0.6,
        base_img_size=224,
        variable_num_frames=False,
    ):
        super().__init__()

        self.model_name = model_name
        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m',
                              'vit_giant_patch14_reg4_dinov2.lvd142m', 'vit_base_patch16_clip_224.openai',
                              "vit_base_patch16_clip_224.laion2b", "samvit_base_patch16.sa1b", "eva02_base_patch16_clip_224.merged2b"], f"{model_name} not found"

        # parameters
        self.img_size = model_kwargs['img_size']
        self.num_frames = model_kwargs['num_frames']
        self.variable_num_frames = variable_num_frames
        self.patch_size = model_kwargs['patch_size']
        self.t_patch_size = model_kwargs['t_patch_size']
        self.num_latent_tokens = num_latent_tokens

        model_kwargs['num_latent_tokens'] = num_latent_tokens

        assert not (variable_num_frames and rope_mixed)

        # load model
        if self.t_patch_size > 1:
            model_kwargs['embed_layer'] = PatchEmbed3D
            self.is_3d_patchify = True
        else:
            self.is_3d_patchify = False

        if not variable_num_frames:
            self.num_img_tokens = (self.img_size * self.img_size * self.num_frames) / (self.patch_size * self.patch_size * self.t_patch_size)

        num_patch = model_kwargs['img_size'] // model_kwargs['patch_size']

        model_kwargs['attn_layer'] = partial(Attention, rope_heads=rope_heads)
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )

        self.embed_dim = model.embed_dim
        # get num of img tokens
        if not variable_num_frames:
            assert self.num_img_tokens == model.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        if self.num_latent_tokens:
            # latent tokens
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)

            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            trunc_normal_(self.latent_pos_embed, std=.02)

            if tuning_method == 'frozen':
                self.latent_tokens.requires_grad = False
                self.latent_pos_embed.requires_grad = False

        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            # self.mask_ratio_generator = stats.truncnorm((1.0 - token_drop) / 0.25, 1.0 / 0.25, loc=1.0, scale=0.25)
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
            nn.init.normal_(self.mask_token, std=.02)

        # rope
        self.use_ape = use_ape
        self.use_rope = use_rope
        # if self.use_rope:
        #     self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.rope_theta_t = rope_theta_t
        self.rope_layers = rope_layers
        if rope_layers is not None:
            print(f'TimmViTEncoder set rope_layers={rope_layers}, only first {rope_layers} layers use RoPE')

        assert not (self.num_frames > 1 and self.rope_mixed)
        
        if self.use_rope:
            if self.rope_mixed:
                self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
                
                freqs = []
                for i, _ in enumerate(model.blocks):
                    freqs.append(
                        init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                    )
                freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
                
                if base_img_size != model_kwargs['img_size']:
                    t_x, t_y = init_t_xy(end_x=num_patch , end_y=num_patch)
                else:
                    t_x, t_y = init_t_xy(end_x=num_patch , end_y=num_patch)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
            else: # we use this
                rope_dim = rope_dim if rope_dim else model.embed_dim//model.num_heads
                self.compute_cis = partial(compute_axial_cis_3d, dim=rope_dim, theta_xy=rope_theta, theta_t=rope_theta_t)
                # self.compute_cis = partial(compute_axial_cis_2d, dim=model.embed_dim//model.num_heads, theta=rope_theta)
                freqs_cis = self.compute_cis(end_x=num_patch, end_y=num_patch, end_t=self.num_frames//self.t_patch_size)
                self.freqs_cis = freqs_cis

        # if not self.use_ape:
        #     for b in self.model.blocks:
        #         b.attn.flash_attn = False

        if self.variable_num_frames:
            self.rope_buffer = {}


    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_pos_embed', 'freqs']

    def sample_orders(self, bsz, seq_len):
        return torch.stack([torch.randperm(seq_len) for _ in range(bsz)], dim=0)

    def random_token_masking(self, x, orders):
        bsz, seq_len = x.size(0), x.size(1)
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=x.device)
        # stats.truncnorm.rvs
        mask_ratios = self.mask_ratio_generator.rvs(size=bsz)
        
        for i in range(bsz):
            ratio = mask_ratios[i]
            num_mask = int(seq_len * ratio)
            indices = orders[i][:num_mask]
            mask[i, indices] = True
            
        return mask
    
    def forward(self, x, return_mask=False, num_frames=None, fps=None, raw_num_frames=None, frame_pts=None):
        """
        当frame_pts设置为None: fps应该设置为采样后的值
        当frame_pts设置为时间戳/帧编号: fps应该设置为采样前的值
        """
        assert not (self.variable_num_frames and (num_frames is None or fps is None))
        assert not (self.t_patch_size > 1 and frame_pts is not None)
        if num_frames is None:
            num_frames = self.num_frames
        if raw_num_frames is None:
            raw_num_frames = num_frames

        # get tokens
        H, W = x.shape[-2:]
        if num_frames > 1:
            assert x.shape[2] == num_frames
            if not self.is_3d_patchify:
                x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.model.patch_embed(x)
        if num_frames > 1 and not self.is_3d_patchify:
            x = rearrange(x, '(b f) n c -> b (f n) c', f=num_frames)

        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_token_masking(x, orders).unsqueeze(-1)
            # print(mask.sum(), mask.shape)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool, device=x.device).unsqueeze(-1)
        
        if not 'eva02' in self.model_name:
            x = self.model._pos_embed(x, use_ape=not self.variable_num_frames)
            x = self.model.patch_drop(x)
        else:
            x, _ = self.model._pos_embed(x, use_ape=not self.variable_num_frames)

        if self.num_latent_tokens:
            # insert latent tokens
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            z = z + self.latent_pos_embed
            x = torch.cat([x, z], dim=1)
            
        # pre layer norm
        if not 'eva02' in self.model_name:
            x = self.model.norm_pre(x)

        if not self.use_rope: #self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
        else:
            if self.rope_mixed:
                if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                    t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                    t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                else:
                    t_x, t_y = self.freqs_t_x, self.freqs_t_y
                freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
                
                for i , blk in enumerate(self.model.blocks):
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
            else:
                if self.variable_num_frames:
                    with torch.no_grad():
                        if fps in self.rope_buffer:
                            freqs_cis = self.rope_buffer[fps].clone().detach()
                        else:
                            freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size, end_t=raw_num_frames//self.t_patch_size, fps=fps, frame_pts=frame_pts)
                            self.rope_buffer[fps] = freqs_cis.clone().detach()
                elif self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                    freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size, end_t=raw_num_frames//self.t_patch_size)
                else:
                    freqs_cis = self.freqs_cis
                freqs_cis = freqs_cis.to(x.device)
                
                for i , blk in enumerate(self.model.blocks):
                    if (self.rope_layers is not None) and (i >= self.rope_layers):
                        x = blk(x)
                    else:
                        x = blk(x, freqs_cis=freqs_cis, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                
        # x = self.model.blocks(x)
        if not 'eva02' in self.model_name:
            x = self.model.norm(x)
        else:
            x = self.model.fc_norm(x)

        if self.num_latent_tokens:
            # get z tokens as out
            out = x[:, -self.num_latent_tokens:]
        else:
            # get img tokens as out
            out = x[:, self.num_prefix_tokens:]

        if return_mask:
            return out, mask.bool()
        else:
            return out

class TimmViTDecoder(nn.Module):
    def __init__(self, 
        in_channels=3,
        model_name='vit_small_patch14_dinov2.lvd142m',
        model_kwargs={'img_size': 224, 'num_frames': 1, 'patch_size': 14, 't_patch_size': 1, 'drop_path_rate': 0.0}, pretrained=True,
        tuning_method='lora', tuning_kwargs={'r': 8},
        num_latent_tokens=32, to_pixel='linear',
        codebook_embed_dim=32,
        rope_theta=100.0, rope_theta_t=100.0, rope_mixed=False, use_rope=False, use_ape=True,
        rope_dim=None, rope_heads=None, rope_layers=None,
        cls_token=True,
        base_img_size=224,
        seperate_mask_token=False,
        variable_num_frames=False,
        use_coord_mlp=False
    ):
        super().__init__()

        self.patch_size = model_kwargs['patch_size']
        self.t_patch_size = model_kwargs['t_patch_size']
        self.variable_num_frames = variable_num_frames
        model_kwargs['num_latent_tokens'] = num_latent_tokens
        if self.t_patch_size > 1:
            model_kwargs['embed_layer'] = PatchEmbed3D
            self.is_3d_patchify = True
        else:
            self.is_3d_patchify = False

        assert not (variable_num_frames and (rope_mixed or seperate_mask_token))

        model_kwargs['attn_layer'] = partial(Attention, rope_heads=rope_heads)
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        
        self.embed_dim = model.embed_dim
        # get num of img tokens
        if not variable_num_frames:
            self.num_img_tokens = model.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # learnable image tokens
        self.seperate_mask_token = seperate_mask_token
        if self.seperate_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, self.num_img_tokens, model.embed_dim))
            print(f'use seperate_mask_token: {self.mask_token.shape}')
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.use_coord_mlp = use_coord_mlp
        if use_coord_mlp:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3 * 10, model.embed_dim),
                nn.ReLU(),
                nn.Linear(model.embed_dim, model.embed_dim),
            )
            self.coord_buffer = {}

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)
        
        self.use_ape = use_ape
        self.use_rope = use_rope
        # if self.use_rope:
        #     self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.num_frames = model_kwargs['num_frames']
        self.rope_theta_t = rope_theta_t
        self.rope_layers = rope_layers
        if rope_layers is not None:
            print(f'TimmViTDecoder set rope_layers={rope_layers}, only last {rope_layers} layers use RoPE')

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], num_frames=self.num_frames, \
                                in_channels=in_channels, in_dim=model.embed_dim, 
                                patch_size=self.patch_size, t_patch_size=self.t_patch_size)

        assert not (self.num_frames > 1 and self.rope_mixed)

        num_patch = model_kwargs['img_size'] // model_kwargs['patch_size']

        if self.use_rope:
            if self.rope_mixed:
                self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
                
                freqs = []
                for i, _ in enumerate(model.blocks):
                    freqs.append(
                        init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                    )
                freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
                
                if base_img_size != model_kwargs['img_size']:
                    t_x, t_y = init_t_xy(end_x=num_patch , end_y=num_patch)
                else:
                    t_x, t_y = init_t_xy(end_x=num_patch, end_y=num_patch)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
            elif not self.rope_mixed: # we use this
                rope_dim = rope_dim if rope_dim else model.embed_dim//model.num_heads
                self.compute_cis = partial(compute_axial_cis_3d, dim=rope_dim, theta_xy=rope_theta, theta_t=rope_theta_t)
                # self.compute_cis = partial(compute_axial_cis_2d, dim=model.embed_dim//model.num_heads, theta=rope_theta)
                
                freqs_cis = self.compute_cis(end_x=num_patch, end_y=num_patch, end_t=self.num_frames//self.t_patch_size)
                self.freqs_cis = freqs_cis

        if self.variable_num_frames:
            self.rope_buffer = {}
            
        # if not self.use_ape:
        #     for b in self.model.blocks:
        #         b.attn.flash_attn = False


        if 'movq' in model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            

        self.cls_token = cls_token
        if not cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()

    @torch.no_grad()
    def make_coord_grid(self, f: int, h: int, w: int, num_freqs: int=10) -> torch.Tensor:
        t_idx = torch.arange(f, dtype=torch.float32)
        y_idx = torch.arange(h, dtype=torch.float32)
        x_idx = torch.arange(w, dtype=torch.float32)

        t_norm = t_idx / f
        y_norm = y_idx / h
        x_norm = x_idx / w

        T, Y, X = torch.meshgrid(t_norm, y_norm, x_norm, indexing='ij')
        coords = torch.stack((T.reshape(-1), X.reshape(-1), Y.reshape(-1)), dim=-1)

        freqs = 2 ** torch.arange(num_freqs, dtype=coords.dtype)
        coords_exp = coords.unsqueeze(-1) * freqs.view(1, 1, -1) * torch.pi
        pe = torch.sin(coords_exp)
        coords = pe.view(-1, 3 * num_freqs)

        return coords

    def forward(self, z, interpolate_zq=None, H=None, W=None, num_frames=None, fps=None, raw_num_frames=None, frame_pts=None):
        assert not (self.variable_num_frames and (num_frames is None or fps is None))
        assert not (self.t_patch_size > 1 and frame_pts is not None)

        F = self.num_frames if num_frames is None else num_frames
        if raw_num_frames is None:
            raw_num_frames = F
        if H is None:
            num_img_tokens = self.num_img_tokens
            H = W = int(math.sqrt(num_img_tokens // F)) * self.patch_size
        else:
            num_img_tokens = H * W * F // self.patch_size ** 2 // self.t_patch_size
        
        if not self.variable_num_frames:
            assert num_img_tokens == self.num_img_tokens

        # mask tokens
        if self.num_latent_tokens:
            if self.seperate_mask_token:
                x = self.mask_token.expand(z.size(0), -1, -1)
            else:
                x = self.mask_token.expand(z.size(0), num_img_tokens, -1)
            if self.use_coord_mlp:
                with torch.no_grad():
                    if (F, H, W) in self.coord_buffer:
                        coord = self.coord_buffer[(F, H, W)].clone().detach()
                    else:
                        coord = self.make_coord_grid(F // self.t_patch_size, H // self.patch_size, W // self.patch_size)
                        coord = coord.to(x.device, dtype=x.dtype)
                        self.coord_buffer[(F, H, W)] = coord.clone().detach()
                coord_emb = self.coord_mlp(coord)
                coord_emb = coord_emb.expand(z.size(0), -1, -1)
                x = x + coord_emb
        else:
            x = z
            
        x = self.model._pos_embed(x, use_ape=self.use_ape)
        x = self.model.patch_drop(x)
        
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.model.norm_pre(x)
        
        
        if not self.use_rope: #self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        else:
            if self.rope_mixed:
                if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                    t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                    t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                else:
                    t_x, t_y = self.freqs_t_x, self.freqs_t_y
                freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
                
                for i , blk in enumerate(self.model.blocks):
                    if self.use_movq:
                        x = blk(x, interpolate_zq, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                    else:
                        x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

            else:
                if self.variable_num_frames:
                    with torch.no_grad():
                        if fps in self.rope_buffer:
                            freqs_cis = self.rope_buffer[fps].clone().detach()
                        else:
                            freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size, end_t=raw_num_frames//self.t_patch_size, fps=fps, frame_pts=frame_pts)
                            self.rope_buffer[fps] = freqs_cis.clone().detach()
                elif self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                    freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size, end_t=F//self.t_patch_size)
                else:
                    freqs_cis = self.freqs_cis
                freqs_cis = freqs_cis.to(x.device)
                
                n = len(self.model.blocks)
                for i , blk in enumerate(self.model.blocks):
                    if self.use_movq:
                        if (self.rope_layers is not None) and (i < n - self.rope_layers):
                            x = blk(x, interpolate_zq)
                        else:
                            x = blk(x, interpolate_zq, freqs_cis=freqs_cis, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                    else:
                        if (self.rope_layers is not None) and (i < n - self.rope_layers):
                            x = blk(x)
                        else:
                            x = blk(x, freqs_cis=freqs_cis, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        x = x[:, self.num_prefix_tokens:-self.num_latent_tokens]

        out = self.to_pixel(x, F)

        return out


if __name__ == '__main__':
    encoder = TimmViTEncoder(num_latent_tokens=256)
    decoder = TimmViTDecoder(num_latent_tokens=256)
    
    x = torch.randn(1, 3, 224, 224)
    
    o = encoder(x)
    print(o.shape)
    r = decoder(o)