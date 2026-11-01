import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleDenseSegmentation.models import backbones

__all__ = [
    'dinov3_vit_small_patch16_universal_segmentation',
    'dinov3_vit_small_plus_patch16_universal_segmentation',
    'dinov3_vit_base_patch16_universal_segmentation',
    'dinov3_vit_large_patch16_universal_segmentation',
    'dinov3_vit_large_plus_patch16_universal_segmentation',
    'dinov3_vit_huge_plus_patch16_universal_segmentation',
]


class ScaleBlock(nn.Module):

    def __init__(self, inplanes):
        super(ScaleBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(inplanes,
                                        inplanes,
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        output_padding=0,
                                        bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(inplanes,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               groups=inplanes,
                               bias=False)
        self.norm = nn.LayerNorm(inplanes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        # 从 [N, C, H, W] 转换为 [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # 转换回 [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        return x


class DINOV3UniversalSegmentation(nn.Module):
    """
    num_classes数量必须包含背景类
    """

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=1024,
                 query_num=200,
                 num_classes=2,
                 query_block_nums=4,
                 use_gradient_checkpoint=False):
        super(DINOV3UniversalSegmentation, self).__init__()
        self.image_size = image_size
        self.query_num = query_num
        self.num_classes = num_classes
        self.query_block_nums = query_block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        embedding_planes = self.backbone.out_channels
        patch_size = self.backbone.patch_size
        self.grid_size = image_size // patch_size
        self.block_nums = len(self.backbone.blocks)

        self.query_embedding = nn.Embedding(query_num, embedding_planes)
        self.class_pred = nn.Linear(embedding_planes, num_classes)

        self.query_proj = nn.Sequential(
            nn.Linear(embedding_planes, embedding_planes), nn.GELU(),
            nn.Linear(embedding_planes, embedding_planes), nn.GELU(),
            nn.Linear(embedding_planes, embedding_planes),
            nn.LayerNorm(embedding_planes))

        upscale_blocks_num = max(1, int(math.log2(patch_size)) - 2)
        self.upscale_blocks = nn.ModuleList(
            [ScaleBlock(embedding_planes) for _ in range(upscale_blocks_num)])

    def predict(self, x):
        # torch.Size([B, query_num, C])
        q = x[:, :self.query_num, :]

        # torch.Size([B, query_num, num_classes])
        class_preds = self.class_pred(q)

        # torch.Size([B, H*W, C])
        x = x[:, self.query_num:, :]
        # torch.Size([B, C, H, W])
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.grid_size,
                                      self.grid_size)

        # q:torch.Size([B, query_num, C])
        q = self.query_proj(q)

        # upscale image features
        for block in self.upscale_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        q = q.float()
        x = x.float()

        # torch.Size([B, query_num, C]) torch.Size([B, C, H', W']) -> torch.Size([B, query_num, H', W'])
        mask_preds = torch.einsum("bqc, bchw -> bqhw", q, x)

        mask_preds = F.interpolate(mask_preds,
                                   (self.image_size, self.image_size),
                                   mode="bilinear")

        return mask_preds, class_preds

    def forward(self, x):
        # x: [B, 3, image_size, image_size]

        x = self.backbone.patch_embed(x)

        # reshape to sequence: [B, H, W, C] -> [B, H*W, C]
        _, H, W, _ = x.shape
        x = x.flatten(1, 2)

        # compute rope sincos embeddings: (H*W, D) (H*W, D)
        rope_sincos = self.backbone.rope_embed(H=H, W=W)

        for idx, block in enumerate(self.backbone.blocks):
            # add query_embedding before query_block_nums blocks
            if idx == self.block_nums - self.query_block_nums:
                query_ids = torch.arange(self.query_num, device=x.device)
                # concatenate query and image tokens: [B, query_num + H*W, C]
                x = torch.cat([
                    self.query_embedding(query_ids)[None].expand(
                        x.shape[0], -1, -1),
                    x,
                ],
                              dim=1)

            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, rope_sincos, use_reentrant=False)
            else:
                x = block(x, rope_sincos)

        # predict masks and classes
        mask_preds, class_preds = self.predict(self.backbone.norm(x))

        return mask_preds, class_preds


def _universal_segmentation(backbone_type, backbone_pretrained_path, **kwargs):
    model = DINOV3UniversalSegmentation(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        **kwargs)

    return model


def dinov3_vit_small_patch16_universal_segmentation(
        backbone_pretrained_path='', **kwargs):
    return _universal_segmentation('dinov3_vit_small_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


def dinov3_vit_small_plus_patch16_universal_segmentation(
        backbone_pretrained_path='', **kwargs):
    return _universal_segmentation('dinov3_vit_small_plus_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


def dinov3_vit_base_patch16_universal_segmentation(backbone_pretrained_path='',
                                                   **kwargs):
    return _universal_segmentation('dinov3_vit_base_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


def dinov3_vit_large_patch16_universal_segmentation(
        backbone_pretrained_path='', **kwargs):
    return _universal_segmentation('dinov3_vit_large_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


def dinov3_vit_large_plus_patch16_universal_segmentation(
        backbone_pretrained_path='', **kwargs):
    return _universal_segmentation('dinov3_vit_large_plus_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


def dinov3_vit_huge_plus_patch16_universal_segmentation(
        backbone_pretrained_path='', **kwargs):
    return _universal_segmentation('dinov3_vit_huge_plus_patch16_backbone',
                                   backbone_pretrained_path, **kwargs)


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net = dinov3_vit_small_patch16_universal_segmentation()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('2222', mask_preds.shape, class_preds.shape)

    net = dinov3_vit_small_patch16_universal_segmentation(
        use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('1111', mask_preds.shape, class_preds.shape)

    net = dinov3_vit_small_plus_patch16_universal_segmentation()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('2222', mask_preds.shape, class_preds.shape)

    net = dinov3_vit_base_patch16_universal_segmentation()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('2222', mask_preds.shape, class_preds.shape)

    net = dinov3_vit_large_patch16_universal_segmentation()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('2222', mask_preds.shape, class_preds.shape)
