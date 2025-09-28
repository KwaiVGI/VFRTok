import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', 
                 img_size=256, num_frames=1, in_channels=3, in_dim=512, 
                 patch_size=16, t_patch_size=1) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.in_channels = in_channels
        self.num_frames = num_frames
        if to_pixel == 'linear':
            self.model = nn.Linear(in_dim, in_channels * t_patch_size * patch_size * patch_size) # b n d -> b n (c tp p p)
        elif to_pixel == 'conv':
            self.num_patches = (img_size // patch_size) ** 2
            assert not num_frames > 1
            num_patches_per_dim = img_size // patch_size  # e.g. 256//16 = 16
            self.model = nn.Sequential(
                # (B, L, C) -> (B, C, H, W) with H = W = num_patches_per_dim
                Rearrange('b (h w) c -> b c h w', h=num_patches_per_dim),
                
                # For example, first reduce dimension via a 1x1 conv from in_dim -> 128
                nn.Conv2d(in_dim, 128, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),

                # Upsample from size (num_patches_per_dim) to a larger intermediate
                nn.Upsample(scale_factor=2, mode='nearest'),  
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                # Repeat upsampling until we reach the final resolution
                # For a 16x16 patch layout, we need 4x upsampling to reach 256
                #   16 -> 32 -> 64 -> 128 -> 256
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            )
        elif to_pixel == 'siren':
            assert not num_frames > 1
            self.model = nn.Sequential(
                SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                SineLayer(in_dim * 2, img_size // patch_size * patch_size * in_channels, is_first=False, omega_0=30)
            )
        elif to_pixel == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_pixel_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x, num_frames):
        """
        For image:
            x: (N, L, patch_size**2 *3)
            imgs: (N, 3, H, W)
        For video:
            x: (N, L, patch_size**2 *3)
            vids: (N, 3, F, H, W)
        """
        p = self.patch_size
        pf = self.t_patch_size
        f = num_frames // pf
        h = w = int((x.shape[1] // f) ** .5)
        assert h * w * f == x.shape[1], print(h, w, f, x.shape[1])
        if f > 1:
            # imgs = rearrange(x, 'b (f h w) (p1 p2 c) -> b c f (h p1) (w p2)', f=f, h=h, w=w, c=3, p1=p, p2=p)
            imgs = rearrange(x, 'b (f h w) (pf ph pw c) -> b c (f pf) (h ph) (w pw)', f=f, h=h, w=w, c=3, pf=pf, ph=p, pw=p)
        else:
            imgs = rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=h, w=w, c=3, ph=p, pw=p)
        return imgs

    def forward(self, x, num_frames=None):
        if num_frames is None:
            num_frames = self.num_frames
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x, num_frames)
        elif self.to_pixel_name == 'siren':
            x = self.model(x)
            x = x.view(x.shape[0], self.in_channels, self.patch_size * int(self.num_patches ** 0.5),
                       self.patch_size * int(self.num_patches ** 0.5))
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            pass
        return x

if __name__ == '__main__':
    x = torch.rand((4, 256, 768))
    h = w = p = 16
    y = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    y = torch.einsum('nhwpqc->nchpwq', y)
    y = y.reshape(shape=(y.shape[0], 3, h * p, h * p))
    z = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, c=3, p1=p, p2=p)
    assert (y == z).all()