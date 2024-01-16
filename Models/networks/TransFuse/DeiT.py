# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from Models.networks.TransFuse.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取补丁数量
        num_patches = self.patch_embed.num_patches
        # 创建一个形状为(1, num_patches + 1, embed_dim)的位置嵌入张量
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, self.embed_dim))

    def forward(self, x):
        # 对输入进行补丁嵌入操作，得到形状为(B, num_patches, embed_dim)的张量
        x = self.patch_embed(x)
        # 获取位置嵌入张量，并加到补丁嵌入张量上
        pe = self.pos_embed
        x = x + pe
        # 进行位置丢弃
        x = self.pos_drop(x)
        # 对补丁嵌入张量进行Transformer块的循环处理
        for blk in self.blocks:
            x = blk(x)
        # 进行归一化操作
        x = self.norm(x)
        return x


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    
    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    # pe = F.interpolate(pe, size=(32, 32), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model