import torch
import torch.nn as nn


def to_2tuple(n):
    return (n, n)


class TokenEmbed(nn.Module):
    """ 2D Image to Token Embedding
    """

    def __init__(
            self,
            img_size: int = 28,
            patch_size: int = 4,
            in_chans: int = 1,
            embed_dim: int = 128,
            norm_layer: bool = None,
            flatten: bool = True,
            output_fmt: str = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x