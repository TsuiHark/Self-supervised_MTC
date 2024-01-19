import torch
from torch import nn
import numpy as np
from modules.TokenEmbedding import TokenEmbed
from modules.PositionEmbedding import get_2d_sincos_pos_embed
from modules.transformer import Block


# encoder
class encoder(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                 encoder_depth=8, num_heads=8, mlp_ratio=3., norm_layer=nn.LayerNorm):
        super(encoder, self).__init__()

        self.token_embed = TokenEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.token_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) encoder_pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.token_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def random_masking(self, x, mask_ratio=0.5):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        patches_mask = torch.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is removed
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0):
        x = self.token_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        z = self.norm(x)
        # x_cls = z[:, :1, :]
        # print(x_cls.shape)  # (bs, 1, 128)

        return z, ids_restore


# decoder
class decoder(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                 decoder_depth=4, num_heads=8, mlp_ratio=3., norm_layer=nn.LayerNorm):
        super(decoder, self).__init__()

        self.token_embed = TokenEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.token_embed.num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(embed_dim)

        # Predictor
        self.predictor = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) encoder_pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.token_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.token_embed.patch_size[0]  # 4
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)

        # imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def forward(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed w/o cls token
        x = x + self.decoder_pos_embed


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        x = self.predictor(x)

        x = self.unpatchify(x)

        return x


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Encoder = encoder()
    Encoder.to(device)

    # summary(model, input_size=(1, 28, 28), batch_size=-1)

    x = torch.randn(1, 1, 28, 28).to(device)
    y, _ = Encoder(x, mask_ratio=0.6)
    print(y.shape)
