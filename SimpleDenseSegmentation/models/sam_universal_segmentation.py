import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

__all__ = [
    'sam_vit_base_patch16_universal_segmentation',
    'sam_vit_large_patch16_universal_segmentation',
    'sam_vit_huge_patch16_universal_segmentation',
]


class PatchEmbed(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=768,
                 kernel_size=16,
                 stride=16,
                 padding=0):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        x = self.proj(x)

        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(rel_pos.reshape(
            1, rel_pos.shape[0], -1).permute(0, 2, 1),
                                        size=max_rel_dist,
                                        mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):

    def __init__(self, inplanes, head_nums=8, input_size=None):
        super(Attention, self).__init__()
        self.head_nums = head_nums
        head_planes = inplanes // head_nums
        self.scale = head_planes**-0.5

        self.qkv = nn.Linear(inplanes, inplanes * 3)
        self.proj = nn.Linear(inplanes, inplanes)

        assert (
            input_size is not None
        ), "Input size must be provided if using relative positional encoding."
        # initialize relative positional embeddings
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * input_size[0] - 1, head_planes))
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * input_size[1] - 1, head_planes))

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.head_nums,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.head_nums, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w,
                                      (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.head_nums, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MLPBlock(nn.Module):

    def __init__(self, inplanes, mlp_planes):
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(inplanes, mlp_planes)
        self.lin2 = nn.Linear(mlp_planes, inplanes)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.lin2(self.act(self.lin1(x)))

        return x


class Block(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 mlp_ratio=4.0,
                 input_size=None,
                 window_size=0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes, eps=1e-6)
        self.attn = Attention(inplanes=inplanes,
                              head_nums=head_nums,
                              input_size=input_size if window_size == 0 else
                              (window_size, window_size))

        self.norm2 = nn.LayerNorm(inplanes, eps=1e-6)
        self.mlp = MLPBlock(inplanes=inplanes,
                            mlp_planes=int(inplanes * mlp_ratio))

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class ViTImageEncoder(nn.Module):

    def __init__(self,
                 image_size=1024,
                 patch_size=16,
                 inplanes=3,
                 embedding_planes=768,
                 block_nums=12,
                 head_nums=12,
                 mlp_ratio=4,
                 window_size=0,
                 global_attn_indexes=(),
                 use_gradient_checkpoint=False):
        super(ViTImageEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = PatchEmbed(inplanes=inplanes,
                                      planes=embedding_planes,
                                      kernel_size=patch_size,
                                      stride=patch_size,
                                      padding=0)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, image_size // patch_size, image_size // patch_size,
                        embedding_planes))

        blocks = []
        for i in range(block_nums):
            block = Block(
                inplanes=embedding_planes,
                head_nums=head_nums,
                mlp_ratio=mlp_ratio,
                input_size=(image_size // patch_size,
                            image_size // patch_size),
                window_size=window_size if i not in global_attn_indexes else 0)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.out_channels = embedding_planes

    def forward(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed

        for block in self.blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return x


class QueryAttention(nn.Module):

    def __init__(self, inplanes, head_nums=8):
        super(QueryAttention, self).__init__()
        self.head_nums = head_nums
        self.qkv = nn.Linear(inplanes, inplanes * 3)
        self.proj = nn.Linear(inplanes, inplanes)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3, head_nums, head_dim] -> [3, B, head_nums, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.head_nums,
                                  -1).permute(2, 0, 3, 1, 4)
        # each: [B, head_nums, N, head_dim]
        q, k, v = qkv.unbind(0)

        # equivalent to: softmax(q @ k^T / sqrt(head_dim)) @ v
        # [B, head_nums, N, head_dim]
        x = F.scaled_dot_product_attention(q, k, v)
        # [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class QueryBlock(nn.Module):

    def __init__(self, inplanes, head_nums=8, mlp_ratio=4.0):
        super(QueryBlock, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes, eps=1e-6)
        self.attn = QueryAttention(inplanes=inplanes, head_nums=head_nums)
        self.norm2 = nn.LayerNorm(inplanes, eps=1e-6)
        self.mlp = MLPBlock(inplanes=inplanes,
                            mlp_planes=int(inplanes * mlp_ratio))

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.attn(x)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


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


class SAMUniversalSegmentation(nn.Module):
    """
    num_classes数量必须包含背景类
    """

    def __init__(self,
                 image_size=1024,
                 patch_size=16,
                 embedding_planes=768,
                 block_nums=12,
                 head_nums=12,
                 mlp_ratio=4,
                 window_size=14,
                 global_attn_indexes=[2, 5, 8, 11],
                 query_num=200,
                 num_classes=2,
                 query_block_nums=4,
                 use_gradient_checkpoint=False):
        super(SAMUniversalSegmentation, self).__init__()
        self.image_size = image_size
        self.query_num = query_num
        self.num_classes = num_classes
        self.query_block_nums = query_block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            inplanes=3,
            embedding_planes=embedding_planes,
            block_nums=block_nums,
            head_nums=head_nums,
            mlp_ratio=mlp_ratio,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            use_gradient_checkpoint=use_gradient_checkpoint)

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

        # transformer blocks for query-image interaction
        self.query_blocks = nn.ModuleList([
            QueryBlock(embedding_planes,
                       head_nums=head_nums,
                       mlp_ratio=mlp_ratio) for _ in range(query_block_nums)
        ])

        # norm for features before prediction
        self.feat_norm = nn.LayerNorm(embedding_planes)

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
        # Called through __call__ (not .forward()) so that frameworks which
        # hook forward to materialise a module's own parameters -- e.g. ZeRO-3
        # parameter partitioning -- see this call. Invoking .forward() directly
        # bypasses those hooks, leaving ViTImageEncoder's own pos_embed as an
        # unmaterialised placeholder.
        x = self.backbone(x)

        B, H, W, C = x.shape

        # reshape to sequence: [B, H, W, C] -> [B, H*W, C]
        x = x.reshape(B, H * W, C)

        # expand query embeddings: [B, query_num, C]
        #
        # The whole table is read through nn.Embedding.forward rather than off
        # self.query_embedding.weight. For an Embedding whose only options are
        # the defaults, forward is a plain row lookup, so asking for rows
        # 0..query_num-1 returns exactly the weight matrix: identical values
        # and identical gradients.
        #
        # Going through forward matters because parameter sharding frameworks
        # hook a module's forward to materialise that module's own parameters;
        # a bare attribute read bypasses the hook and can hand back a
        # placeholder. This keeps the model a plain nn.Module while staying
        # correct under such frameworks.
        query_ids = torch.arange(self.query_num, device=x.device)
        q = self.query_embedding(query_ids)[None].expand(B, -1, -1)

        # concatenate query and image tokens: [B, query_num + H*W, C]
        x = torch.cat([q, x], dim=1)

        for block in self.query_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # predict masks and classes
        mask_preds, class_preds = self.predict(self.feat_norm(x))

        return mask_preds, class_preds


def _sam_universal_segmentation(image_size, patch_size, embedding_planes,
                                block_nums, head_nums, mlp_ratio, window_size,
                                global_attn_indexes, **kwargs):
    model = SAMUniversalSegmentation(image_size=image_size,
                                     patch_size=patch_size,
                                     embedding_planes=embedding_planes,
                                     block_nums=block_nums,
                                     head_nums=head_nums,
                                     mlp_ratio=mlp_ratio,
                                     window_size=window_size,
                                     global_attn_indexes=global_attn_indexes,
                                     **kwargs)

    return model


def sam_vit_base_patch16_universal_segmentation(image_size=1024,
                                                patch_size=16,
                                                **kwargs):
    return _sam_universal_segmentation(image_size=image_size,
                                       patch_size=patch_size,
                                       embedding_planes=768,
                                       block_nums=12,
                                       head_nums=12,
                                       mlp_ratio=4,
                                       window_size=14,
                                       global_attn_indexes=[2, 5, 8, 11],
                                       **kwargs)


def sam_vit_large_patch16_universal_segmentation(image_size=1024,
                                                 patch_size=16,
                                                 **kwargs):
    return _sam_universal_segmentation(image_size=image_size,
                                       patch_size=patch_size,
                                       embedding_planes=1024,
                                       block_nums=24,
                                       head_nums=16,
                                       mlp_ratio=4,
                                       window_size=14,
                                       global_attn_indexes=[5, 11, 17, 23],
                                       **kwargs)


def sam_vit_huge_patch16_universal_segmentation(image_size=1024,
                                                patch_size=16,
                                                **kwargs):
    return _sam_universal_segmentation(image_size=image_size,
                                       patch_size=patch_size,
                                       embedding_planes=1280,
                                       block_nums=32,
                                       head_nums=16,
                                       mlp_ratio=4,
                                       window_size=14,
                                       global_attn_indexes=[7, 15, 23, 31],
                                       **kwargs)


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

    net = sam_vit_base_patch16_universal_segmentation()
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

    net = sam_vit_base_patch16_universal_segmentation(
        use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    mask_preds, class_preds = net(
        torch.autograd.Variable(torch.rand(1, 3, image_h, image_w)))
    print('2222', mask_preds.shape, class_preds.shape)
