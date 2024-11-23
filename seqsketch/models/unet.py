from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqsketch.models import (
    DownSample,
    ResBlock,
    Swish,
    TimeEmbedding,
    UpSample,
    ImageEmbedding,
)
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        image_resolution = params.image_resolution
        ch = params.ch
        ch_mult = params.ch_mult
        num_res_blocks = params.num_res_blocks
        attn = params.attn
        dropout = params.dropout
        cfg_dropout = params.cfg_dropout
        input_channels = params.input_channels

        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.time_embedding = TimeEmbedding(tdim)

        # classifier-free guidance
        self.cfg_dropout = cfg_dropout
        # conditional class embedding for image
        self.image_embedding = ImageEmbedding(
            output_dim=tdim, image_dim=(image_resolution, image_resolution)
        )
        self.combined_embedding = nn.Linear(2 * tdim, tdim)

        self.head = nn.Conv2d(input_channels, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, input_channels, 3, stride=1, padding=1),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, conditioning_image=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        # if self.training:
        #     # random null conditioning in CFG training.
        #     # conditioning_image: [B, 1, H, W]
        #     # sample first the images in the batch that should be null conditioned
        #     null_samples = torch.randn(conditioning_image.size(0)) < self.cfg_dropout
        #     # make sure all images in the batch are null conditioned (all zeros)
        #     conditioning_image[null_samples] = 0

        # Image embedding
        if conditioning_image is not None:
            imemb = self.image_embedding(conditioning_image)
            emb = torch.cat([temb, imemb], dim=-1)
            emb = self.combined_embedding(emb)
        else:
            emb = temb
        #######################

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, emb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, emb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, emb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
