import math
import types

from transformers import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torchvision.ops import roi_align

__all__ = [
    'qwen35_sam_vit_base_patch16_prompt_segmentation',
    'qwen35_sam_vit_large_patch16_prompt_segmentation',
    'qwen35_sam_vit_huge_patch16_prompt_segmentation',
]


class MaskRegionEncoder(nn.Module):

    def __init__(self,
                 inplanes,
                 out_planes,
                 grid_size=3,
                 num_geo_features=11,
                 num_prompt_types=3,
                 geo_fourier_bands=4,
                 roi_sampling_ratio=2):
        super(MaskRegionEncoder, self).__init__()
        self.inplanes = inplanes
        self.out_planes = out_planes
        self.grid_size = grid_size
        self.num_geo_features = num_geo_features
        self.geo_fourier_bands = geo_fourier_bands
        self.roi_sampling_ratio = roi_sampling_ratio
        # 1 global + grid_size^2 shape + 1 contour + 1 geometry
        self.num_tokens = 1 + grid_size * grid_size + 1 + 1

        # Fuse RoI-aligned appearance with per-cell coverage.
        # Input layout: [feat, feat * cover, cover, cover^2]
        self.grid_fuse = nn.Sequential(
            nn.Linear(inplanes * 2 + 2, inplanes),
            nn.GELU(),
            nn.Linear(inplanes, inplanes),
        )
        # Learnable position embedding for each cell of the shape grid.
        self.cell_pos_embed = nn.Parameter(
            torch.zeros(1, grid_size * grid_size, inplanes))

        geo_planes = self.num_geo_features * (1 + 2 * geo_fourier_bands)
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_planes, inplanes),
            nn.GELU(),
            nn.Linear(inplanes, inplanes),
        )

        # Token role identity. This is what makes the token set ordered and
        # addressable rather than a permutation-invariant bag.
        self.role_embed = nn.Embedding(self.num_tokens, inplanes)
        # Visual prompt type identity (point / box / mask).
        self.type_embed = nn.Embedding(num_prompt_types, inplanes)
        # Used for padded (non-existent) region slots.
        self.empty_embed = nn.Embedding(1, inplanes)

        self.out_proj = nn.Sequential(
            nn.Linear(inplanes, out_planes),
            nn.GELU(),
            nn.Linear(out_planes, out_planes),
            nn.LayerNorm(out_planes),
        )

        nn.init.trunc_normal_(self.cell_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.role_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.empty_embed.weight, std=0.02)

    def pool_masks(self, masks, H, W):
        Hm, Wm = masks.shape[-2], masks.shape[-1]
        assert Hm % H == 0 and Wm % W == 0
        rh, rw = Hm // H, Wm // W

        cover = F.avg_pool2d(masks, kernel_size=(rh, rw),
                             stride=(rh, rw)).squeeze(1)
        occ = F.max_pool2d(masks, kernel_size=(rh, rw),
                           stride=(rh, rw)).squeeze(1)
        # Prompt masks are binary, but be defensive about soft inputs.
        occ = (occ > 0).to(cover.dtype)

        return cover, occ

    def compute_bboxes(self, cover_pad, valid):
        B, N, H, W = cover_pad.shape
        device = cover_pad.device
        dtype = cover_pad.dtype

        # [B, N, W] / [B, N, H]
        col_any = cover_pad.amax(dim=2) > 0
        row_any = cover_pad.amax(dim=3) > 0

        iw = torch.arange(W, device=device, dtype=dtype)
        ih = torch.arange(H, device=device, dtype=dtype)

        big_w = torch.full_like(iw, float(W))
        neg_w = torch.full_like(iw, -1.0)
        big_h = torch.full_like(ih, float(H))
        neg_h = torch.full_like(ih, -1.0)

        x0 = torch.where(col_any, iw, big_w).amin(dim=-1)
        x1 = torch.where(col_any, iw, neg_w).amax(dim=-1) + 1.0
        y0 = torch.where(row_any, ih, big_h).amin(dim=-1)
        y1 = torch.where(row_any, ih, neg_h).amax(dim=-1) + 1.0

        # Empty / padded slots fall back to the full image box so that
        # roi_align always receives a legal region.
        full_x0 = torch.zeros_like(x0)
        full_y0 = torch.zeros_like(y0)
        full_x1 = torch.full_like(x1, float(W))
        full_y1 = torch.full_like(y1, float(H))

        keep = valid > 0
        x0 = torch.where(keep, x0, full_x0)
        y0 = torch.where(keep, y0, full_y0)
        x1 = torch.where(keep, x1, full_x1)
        y1 = torch.where(keep, y1, full_y1)

        # Guarantee a strictly positive extent.
        x1 = torch.maximum(x1, x0 + 1.0)
        y1 = torch.maximum(y1, y0 + 1.0)

        return x0, y0, x1, y1

    def build_geometry(self, weights, weight_sum, edge, x0, y0, x1, y1,
                       valid_h, valid_w, H, W):
        device = weights.device
        dtype = weights.dtype
        B, N, P = weights.shape

        gy = torch.arange(H, device=device,
                          dtype=dtype).view(H, 1).expand(H, W).reshape(P, 1)
        gx = torch.arange(W, device=device,
                          dtype=dtype).view(1, W).expand(H, W).reshape(P, 1)

        area = weight_sum.squeeze(-1)
        safe_area = area.clamp_min(1e-6)

        cy = torch.bmm(weights,
                       gy.unsqueeze(0).expand(B, -1,
                                              -1)).squeeze(-1) / safe_area
        cx = torch.bmm(weights,
                       gx.unsqueeze(0).expand(B, -1,
                                              -1)).squeeze(-1) / safe_area

        vh = valid_h.clamp_min(1e-6)
        vw = valid_w.clamp_min(1e-6)
        valid_area = (vh * vw).clamp_min(1e-6)

        bw = (x1 - x0).clamp_min(1e-6)
        bh = (y1 - y0).clamp_min(1e-6)

        area_ratio = (area / valid_area).clamp(min=1e-8, max=1.0)
        perimeter = edge.flatten(2).sum(-1)

        # Every feature below is kept inside [-1, 1]. ``fourier_encode``
        # multiplies its input by frequencies up to pi * 2**(bands-1), so an
        # out-of-range input aliases into noise and the geometry token stops
        # carrying any usable signal.
        #
        # The coordinate ratios are clamped because ``valid_h``/``valid_w``
        # describe the *non-padded* extent of a letterboxed image while the
        # boxes are measured on the full padded feature map, so a region that
        # touches the padding (and every empty slot, which falls back to the
        # full-image box) yields a ratio slightly above 1.
        #
        # The three intrinsically unbounded quantities (log area ratio, aspect
        # ratio and perimeter/sqrt(area)) are additionally log-compressed.
        geo = torch.stack(
            [
                (cx / vw).clamp(0.0, 1.0),
                (cy / vh).clamp(0.0, 1.0),
                (x0 / vw).clamp(0.0, 1.0),
                (y0 / vh).clamp(0.0, 1.0),
                (x1 / vw).clamp(0.0, 1.0),
                (y1 / vh).clamp(0.0, 1.0),
                (torch.log(area_ratio) / 20.0).clamp(-1.0, 1.0),
                torch.sqrt(area_ratio),
                (torch.log(bw / bh) / 4.0).clamp(-1.0, 1.0),
                (area / (bw * bh).clamp_min(1e-6)).clamp(0.0, 1.0),
                (torch.log1p(perimeter / safe_area.sqrt()) / 5.0).clamp(
                    -1.0, 1.0),
            ],
            dim=-1,
        )

        return geo

    def fourier_encode(self, x, num_bands=8):
        bands = torch.arange(num_bands, device=x.device, dtype=x.dtype)
        freqs = torch.pi * torch.pow(2.0, bands)
        # [..., D, num_bands]
        scaled = x.unsqueeze(-1) * freqs

        encoded = torch.cat([
            x,
            torch.sin(scaled).flatten(-2),
            torch.cos(scaled).flatten(-2),
        ],
                            dim=-1)

        return encoded

    def forward(self,
                backbone_features,
                vprompt_masks,
                valid_sizes=None,
                prompt_type_id=None):
        B, C, H, W = backbone_features.shape
        device = backbone_features.device
        dtype = backbone_features.dtype
        k = self.grid_size
        P = H * W

        # Reading ``.shape`` never forces a device sync, unlike ``nonzero``.
        counts = [int(m.shape[0]) for m in vprompt_masks]
        M = sum(counts)
        N_max = max(counts) if M > 0 else 1

        if M == 0:
            return [
                backbone_features.new_zeros(0, self.num_tokens,
                                            self.out_planes) for _ in range(B)
            ]

        # Masks are pooled and scattered in float32 once. The geometry branch
        # needs that precision: ``cover`` is an average over rh*rw pixels and
        # the geometry features accumulate it over all H*W patches, so bf16
        # (8-bit mantissa) loses ~1e-3 of the area and ~0.01 patch of the
        # centroid. That is the same reason ``roi_align`` below is fed float32.
        masks = torch.cat(
            [m.to(device=device, dtype=torch.float32) for m in vprompt_masks],
            dim=0).unsqueeze(1)
        cover32, occ32 = self.pool_masks(masks, H, W)

        counts_tensor = torch.tensor(counts, device=device, dtype=torch.long)
        batch_index = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.long), counts_tensor)
        slot_index = torch.cat(
            [torch.arange(n, device=device, dtype=torch.long) for n in counts])

        cover_pad32 = masks.new_zeros(B, N_max, H, W)
        occ_pad32 = masks.new_zeros(B, N_max, H, W)
        valid32 = masks.new_zeros(B, N_max)
        cover_pad32[batch_index, slot_index] = cover32
        occ_pad32[batch_index, slot_index] = occ32
        valid32[batch_index, slot_index] = 1.0

        # The appearance branch works in the autocast dtype and only needs a
        # downcast copy. ``.to(dtype)`` is a no-op that returns the very same
        # tensor when the model already runs in float32, so no dtype branch is
        # needed here. Scattering before or after the cast is equivalent: the
        # (batch_index, slot_index) pairs are unique, so the scatter merely
        # moves values around without any arithmetic.
        cover_pad = cover_pad32.to(dtype)
        occ_pad = occ_pad32.to(dtype)
        valid = valid32.to(dtype)

        # [B, C, P]
        feat_flat = backbone_features.flatten(2)

        # ---- (a) global appearance token ----
        weights = cover_pad.flatten(2)
        weight_sum = weights.sum(-1, keepdim=True)
        # Fall back to the (guaranteed non-empty) support if coverage is zero.
        weights = torch.where(weight_sum > 0, weights, occ_pad.flatten(2))
        weight_sum = weights.sum(-1, keepdim=True).clamp_min(1e-6)
        # [B, N, C]
        global_token = torch.bmm(weights, feat_flat.transpose(1,
                                                              2)) / weight_sum

        # ---- (b) shape grid tokens ----
        # Boxes are derived in float32: they feed both roi_align (which needs
        # float32 anyway) and the geometry features.
        x0, y0, x1, y1 = self.compute_bboxes(cover_pad32, valid32)

        flat_x0 = x0[batch_index, slot_index]
        flat_y0 = y0[batch_index, slot_index]
        flat_x1 = x1[batch_index, slot_index]
        flat_y1 = y1[batch_index, slot_index]

        # roi_align expects boxes as [batch_index, x0, y0, x1, y1].
        #
        # RoI sampling is always run in float32: under autocast the feature
        # map would be bf16, whose 8-bit mantissa cannot represent the
        # fractional bin centres of a 64-wide grid accurately enough (the
        # quantisation step reaches ~0.25 patch near the right/bottom edge).
        # Since the whole point of moving to roi_align is sub-patch accurate
        # alignment, the sampling itself must not be the bottleneck.
        feat_boxes = torch.stack([
            batch_index.float(),
            flat_x0.float(),
            flat_y0.float(),
            flat_x1.float(),
            flat_y1.float(),
        ],
                                 dim=1)
        # ``aligned=True`` implements the standard half-pixel correction, so
        # there is no hand-rolled normalisation that can disagree with the
        # sampling convention (the previous grid_sample call did).
        feat_roi = roi_align(backbone_features.float(),
                             feat_boxes,
                             output_size=(k, k),
                             spatial_scale=1.0,
                             sampling_ratio=self.roi_sampling_ratio,
                             aligned=True).to(dtype)

        # Per-cell occupancy is sampled from ``cover32`` (the exact per-patch
        # coverage ratio), not from the binary support ``occ``, hence the
        # cover_* naming: every value below is a coverage ratio in [0, 1].
        # The float32 tensor is passed straight in rather than being routed
        # through the autocast dtype and back, which would add a pointless
        # rounding step to a branch that deliberately samples in float32.
        cover_boxes = feat_boxes.clone()
        cover_boxes[:, 0] = torch.arange(M, device=device, dtype=torch.float32)
        cover_roi = roi_align(cover32.unsqueeze(1),
                              cover_boxes,
                              output_size=(k, k),
                              spatial_scale=1.0,
                              sampling_ratio=self.roi_sampling_ratio,
                              aligned=True).to(dtype)

        # [M, k*k, C] / [M, k*k, 1]
        roi_feat = feat_roi.flatten(2).transpose(1, 2)
        roi_cover = cover_roi.flatten(2).transpose(1, 2).clamp(0.0, 1.0)
        grid_token = self.grid_fuse(
            torch.cat(
                [roi_feat, roi_feat * roi_cover, roi_cover,
                 roi_cover.pow(2)],
                dim=-1))
        grid_token = grid_token + self.cell_pos_embed.to(dtype)

        grid_token_pad = backbone_features.new_zeros(B, N_max, k * k, C)
        grid_token_pad[batch_index, slot_index] = grid_token

        # ---- (c) contour token ----
        # Computed in float32 because the perimeter (a sum over H*W patches)
        # is one of the geometry features; the appearance-side contour token
        # then uses the downcast copy.
        occ_view32 = occ_pad32.reshape(B * N_max, 1, H, W)
        dilated32 = F.max_pool2d(occ_view32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        eroded32 = -F.max_pool2d(
            -occ_view32, kernel_size=3, stride=1, padding=1)
        edge32 = (dilated32 - eroded32).reshape(B, N_max, H, W).clamp_min(0.0)
        edge = edge32.to(dtype)

        edge_weights = edge.flatten(2)
        edge_sum = edge_weights.sum(-1, keepdim=True)
        edge_weights = torch.where(edge_sum > 0, edge_weights, weights)
        edge_sum = edge_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        contour_token = torch.bmm(edge_weights, feat_flat.transpose(
            1, 2)) / edge_sum

        # ---- (d) geometry token ----
        if valid_sizes is None:
            valid_h = torch.full((B, 1),
                                 float(H),
                                 device=device,
                                 dtype=torch.float32)
            valid_w = torch.full((B, 1),
                                 float(W),
                                 device=device,
                                 dtype=torch.float32)
        else:
            if not torch.is_tensor(valid_sizes):
                valid_sizes = torch.as_tensor(valid_sizes, device=device)
            valid_sizes = valid_sizes.to(device=device, dtype=torch.float32)
            scale_h = masks.shape[-2] / H
            scale_w = masks.shape[-1] / W
            valid_h = valid_sizes[:, 0:1] / scale_h
            valid_w = valid_sizes[:, 1:2] / scale_w

        # float32 coverage weights, mirroring the (a) branch but without the
        # precision loss that would corrupt the area/centroid features.
        geo_weights = cover_pad32.flatten(2)
        geo_weight_sum = geo_weights.sum(-1, keepdim=True)
        geo_weights = torch.where(geo_weight_sum > 0, geo_weights,
                                  occ_pad32.flatten(2))
        geo_weight_sum = geo_weights.sum(-1, keepdim=True).clamp_min(1e-6)

        # autocast is disabled for the whole geometry branch: it would
        # otherwise downcast the internal ``torch.bmm`` (used for the centroid)
        # to bf16 regardless of the float32 inputs, which is exactly the
        # precision loss this branch is built to avoid. Only the projected
        # result rejoins the autocast dtype.
        #
        # Because autocast is off, nothing casts the inputs of ``geo_mlp`` to
        # the dtype of its weights any more. That matters when the weights are
        # not float32: DeepSpeed's bf16 mode casts *every* module parameter to
        # bf16 in-place (engine._cast_module_mixed_precision), so a float32
        # activation would hit a bf16 nn.Linear and raise
        # "mat1 and mat2 must have the same dtype". The geometry features are
        # therefore built in float32 (for the precision reasons above) and only
        # cast to the parameter dtype immediately before the projection.
        geo_mlp_dtype = self.geo_mlp[0].weight.dtype
        with torch.autocast(device_type=backbone_features.device.type,
                            enabled=False):
            geo = self.build_geometry(geo_weights, geo_weight_sum, edge32, x0,
                                      y0, x1, y1, valid_h, valid_w, H, W)
            geo_token = self.geo_mlp(
                self.fourier_encode(geo,
                                    self.geo_fourier_bands).to(geo_mlp_dtype))
        geo_token = geo_token.to(dtype)

        # ---- assemble ----
        tokens = torch.cat([
            global_token.unsqueeze(2),
            grid_token_pad,
            contour_token.unsqueeze(2),
            geo_token.unsqueeze(2),
        ],
                           dim=2)

        # The whole table is read through nn.Embedding.forward rather than
        # off .weight. For an Embedding left on its default options, forward is
        # a plain row lookup, so asking for rows 0..num_tokens-1 returns
        # exactly the weight matrix: identical values, identical gradients.
        #
        # Going through forward matters because parameter sharding frameworks
        # hook a module's forward to materialise that module's own parameters;
        # a bare attribute read bypasses the hook and can hand back a
        # placeholder. This keeps the module a plain nn.Module while staying
        # correct under such frameworks.
        role_ids = torch.arange(self.num_tokens, device=device)
        tokens = tokens + self.role_embed(role_ids).to(dtype)[None, None]

        if prompt_type_id is None:
            type_ids = torch.zeros(B, device=device, dtype=torch.long)
        elif torch.is_tensor(prompt_type_id):
            type_ids = prompt_type_id.to(device=device,
                                         dtype=torch.long).reshape(-1)
            if type_ids.numel() == 1:
                type_ids = type_ids.expand(B)
        else:
            type_ids = torch.full((B, ),
                                  int(prompt_type_id),
                                  device=device,
                                  dtype=torch.long)
        tokens = tokens + self.type_embed(type_ids).to(dtype)[:, None, None]

        # Padded slots get a dedicated learnable embedding instead of the
        # meaningless corner feature the old sampler produced.
        empty_ids = torch.zeros(1, dtype=torch.long, device=device)
        tokens = torch.where(valid[..., None, None] > 0, tokens,
                             self.empty_embed(empty_ids).to(dtype)[None, None])

        tokens = self.out_proj(tokens)

        return [tokens[i, :counts[i]] for i in range(B)]


class RegionReadout(nn.Module):

    def __init__(self, planes):
        super(RegionReadout, self).__init__()
        self.planes = planes

        self.query = nn.Parameter(torch.randn(planes) * (planes**-0.5))
        self.key = nn.Linear(planes, planes)
        self.value = nn.Linear(planes, planes)

    def forward(self, embeds, ids, num_slots):
        B, L, D = embeds.shape

        span = (ids >= 0)
        # [B, num_slots, L]
        onehot = F.one_hot(ids.clamp_min(0),
                           num_slots).permute(0, 2, 1).to(embeds.dtype)
        onehot = onehot * span.unsqueeze(1).to(embeds.dtype)

        # [B, L]
        score = torch.einsum('bld,d->bl', self.key(embeds),
                             self.query.to(embeds.dtype)) / (D**0.5)
        # [B, num_slots, L]
        score = score.unsqueeze(1).masked_fill(onehot == 0, float(-1e4))
        weight = score.softmax(dim=-1)

        valid = (onehot.sum(dim=-1) > 0).to(embeds.dtype)
        weight = weight * valid.unsqueeze(-1)

        pooled = torch.bmm(weight, self.value(embeds))

        return pooled, valid


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


class VLMProjector(nn.Module):

    def __init__(self, vlm_hidden_size, seg_hidden_size):
        super(VLMProjector, self).__init__()
        self.proj = nn.Sequential(nn.Linear(vlm_hidden_size, seg_hidden_size),
                                  nn.GELU(),
                                  nn.Linear(seg_hidden_size, seg_hidden_size))

    def forward(self, x):
        x = self.proj(x)

        return x


class ClassPredictor(nn.Module):

    def __init__(self):
        super(ClassPredictor, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor([math.log(1 / 0.07)]),
                                        requires_grad=True)

    def forward(self, query_embeds, cond_embeds, embed_masks=None):
        cls_pred = self.logit_scale.exp() * torch.einsum(
            "bqd,bcd->bqc", F.normalize(query_embeds, dim=-1),
            F.normalize(cond_embeds, dim=-1))
        cls_pred = torch.clamp(cls_pred, min=-50, max=50)

        if embed_masks is not None:
            if embed_masks.ndim == 2:
                embed_masks = embed_masks.unsqueeze(1)
            cls_pred = cls_pred.masked_fill(~embed_masks.bool(), -1e6)

        return cls_pred


class QWEN35SAMPromptSegmentation(nn.Module):

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
                 query_block_nums=4,
                 max_prompt_num=1,
                 region_grid_size=3,
                 vlm_model_path=None,
                 tokenizer_vocab_size=None,
                 use_gradient_checkpoint=False):
        super(QWEN35SAMPromptSegmentation, self).__init__()
        assert vlm_model_path is not None, "vlm_model_path must be provided for prompt segmentation"
        assert query_num % max_prompt_num == 0, f'query_num ({query_num}) must be divisible by max_prompt_num'

        self.image_size = image_size
        self.query_num = query_num
        self.query_block_nums = query_block_nums
        self.max_prompt_num = max_prompt_num
        self.queries_per_slot = query_num // max_prompt_num
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

        self.query_proj = nn.Sequential(
            nn.Linear(embedding_planes, embedding_planes), nn.GELU(),
            nn.Linear(embedding_planes, embedding_planes), nn.GELU(),
            nn.Linear(embedding_planes, embedding_planes),
            nn.LayerNorm(embedding_planes))

        # transformer blocks for query-image-text interaction
        self.query_blocks = nn.ModuleList([
            QueryBlock(embedding_planes,
                       head_nums=head_nums,
                       mlp_ratio=mlp_ratio) for _ in range(query_block_nums)
        ])

        self.feat_norm = nn.LayerNorm(embedding_planes)

        upscale_blocks_num = max(1, int(math.log2(patch_size)) - 2)
        self.upscale_blocks = nn.ModuleList(
            [ScaleBlock(embedding_planes) for _ in range(upscale_blocks_num)])

        self.class_pred = ClassPredictor()
        self.bg_embeds = nn.Embedding(1, embedding_planes)

        self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
            vlm_model_path,
            dtype=torch.bfloat16,
            attn_implementation='flash_attention_2')

        # Resize VLM token embeddings to match tokenizer vocab size.
        # The Qwen35SegTokenizer adds special tokens (<SEG>, <p>, </p>,
        # <region>) which increases the vocab size beyond the pretrained
        # VLM's embedding matrix. Without resizing, input_ids containing
        # these new token ids will cause IndexError.
        if tokenizer_vocab_size is not None:
            current_vocab_size = self.vlm.config.text_config.vocab_size
            if tokenizer_vocab_size > current_vocab_size:
                self.vlm.resize_token_embeddings(tokenizer_vocab_size)
                print(f"VLM token embeddings resized: "
                      f"{current_vocab_size} -> {tokenizer_vocab_size}")

        vlm_hidden_size = self.vlm.config.text_config.hidden_size

        self.mlm_projector = VLMProjector(vlm_hidden_size, embedding_planes)
        # MaskRegionEncoder outputs features in VLM hidden size so they can
        # be scattered into the VLM input sequence at the <region>
        # placeholder positions emitted by the tokenizer.
        self.region_encoder = MaskRegionEncoder(embedding_planes,
                                                vlm_hidden_size,
                                                grid_size=region_grid_size)

        # Attention-pooling readouts over the VLM hidden states. These
        # replace the previous per-sample Python loops that used
        # torch.unique (a device sync) plus a uniform mean.
        self.cond_readout = RegionReadout(embedding_planes)
        self.seg_readout = RegionReadout(embedding_planes)

        # Project the per-prompt condition/segment embeddings into the query
        # space. Query slot k is conditioned on prompt k, which gives the
        # mask decoder a direct, differentiable link to the requested target
        # instead of broadcasting one shared signal to every query.
        self.cond_to_query = nn.Linear(embedding_planes, embedding_planes)
        self.seg_to_query = nn.Linear(embedding_planes, embedding_planes)
        # Used for slots that have no corresponding prompt in this sample.
        self.null_slot = nn.Embedding(1, embedding_planes)
        # These three tensors form the ONLY path by which the prompt identity
        # reaches the mask branch, so they must not start at exactly zero.
        # With zero weights and a zero null_slot the injected term is
        # identically 0 for every slot, which means at step 0 a query
        # conditioned on prompt k, a query conditioned on prompt j and a query
        # belonging to an unused slot are numerically indistinguishable. The
        # slot-constrained matcher then has no signal to anchor slot k to
        # prompt k, and the symmetry has to be broken purely by gradients.
        # Small non-zero initialisation keeps the injection a gentle
        # perturbation of query_embedding while making the slots distinct from
        # the very first step.
        nn.init.trunc_normal_(self.cond_to_query.weight, std=0.02)
        nn.init.zeros_(self.cond_to_query.bias)
        nn.init.trunc_normal_(self.seg_to_query.weight, std=0.02)
        nn.init.zeros_(self.seg_to_query.bias)
        nn.init.trunc_normal_(self.null_slot.weight, std=0.02)

        # Enable gradient checkpointing for VLM to reduce memory usage.
        # Must be done BEFORE freezing parameters, because
        # enable_input_require_grads() registers a forward hook on the
        # embedding layer that forces output.requires_grad = True.
        # This is necessary for gradient checkpointing to work correctly
        # even when VLM parameters are frozen (requires_grad=False):
        # without it, the checkpointed layers would see no gradient-requiring
        # inputs and skip saving activations, breaking the gradient chain
        # to upstream trainable modules (region_encoder, mlm_projector).
        if use_gradient_checkpoint:
            self.vlm.enable_input_require_grads()
            self.vlm.gradient_checkpointing_enable({"use_reentrant": False})

        # Always apply LoRA to VLM language model for parameter-efficient
        # fine-tuning. The LoRA adapters are applied to the token mixing and
        # MLP projection layers of the language model, while the vision
        # encoder ("visual") is excluded via exclude_modules.
        #
        # Qwen3.5 is a hybrid model: its `layer_types` alternate between
        # "full_attention" (Qwen3_5Attention, with q/k/v/o_proj) and
        # "linear_attention" (Qwen3_5GatedDeltaNet, with in_proj_qkv /
        # in_proj_z / in_proj_b / in_proj_a / out_proj). For the 4B config
        # only 8 of the 32 layers are full attention while 24 are linear
        # attention, so targeting the q/k/v/o_proj names alone would leave
        # three quarters of the token mixing layers completely frozen.
        # Both naming families are therefore listed below.
        #
        # Note that `out_proj` is unique to the language side here: the
        # vision tower uses `qkv`/`proj`/`linear_fc1`/`linear_fc2`, and is
        # excluded by `exclude_modules` regardless.
        vlm_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                # full_attention layers (Qwen3_5Attention)
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # linear_attention layers (Qwen3_5GatedDeltaNet)
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                # MLP of every layer
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            exclude_modules=r".*visual.*",
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(self.vlm, vlm_lora_config)

        # Freeze VLM vision encoder explicitly (exclude_modules in LoRA
        # config should already exclude "visual", but we ensure it).
        # After get_peft_model, self.vlm is a PeftModel. The path is:
        #   PeftModel.base_model (LoraModel) -> .model (Qwen3_5ForConditionalGeneration)
        #   -> .model (Qwen3_5Model) -> .visual (Qwen3_5VisionModel)
        visual = self.vlm.base_model.model.model.visual
        visual.requires_grad_(False)

        # merger projects vision encoder output to LLM dimension;
        # Qwen3.5 does NOT have deepstack_merger_list,
        # so only merger needs to remain trainable.
        visual.merger.requires_grad_(True)

        # Make embed_tokens and lm_head trainable so that newly added
        # special tokens (<SEG>, <p>, </p>, <region>) can be learned.
        # After get_peft_model, non-LoRA modules default to frozen.
        vlm_base = self.vlm.base_model.model
        vlm_base.lm_head.requires_grad_(True)
        vlm_base.model.language_model.embed_tokens.requires_grad_(True)

        # Keep every trainable VLM weight in fp32.
        #
        # The VLM is loaded with dtype=torch.bfloat16, and
        # requires_grad_(True) does not change a tensor's dtype. bf16 carries
        # only an 8-bit mantissa, so its smallest representable relative step
        # is ~7.8e-3: an in-place optimizer step of lr=1e-4 (let alone the
        # min_lr=5e-6 the cosine schedule decays to) on a bf16 weight of
        # magnitude ~1 is rounded straight back to the original value. The
        # modules below would therefore barely move, which silently defeats
        # the whole point of unfreezing them -- in particular the newly added
        # special tokens (<SEG>, <p>, </p>, <region>) would never be learned,
        # even though the segmentation conditioning depends entirely on them.
        #
        # The LoRA adapters need no such treatment: peft creates them as fresh
        # fp32 nn.Linear layers.
        #
        # Casting these modules is safe:
        #   - autocast still runs the forward itself in bf16, so throughput is
        #     unchanged;
        #   - `visual.dtype` reads the *first* floating point parameter
        #     (patch_embed, still bf16), so the pixel_values cast inside
        #     get_image_features is unaffected;
        #   - lm_head and embed_tokens are weight-tied, and `.float()` casts
        #     the shared tensor in place, so they stay tied;

        #
        # Qwen3.5 has no deepstack_merger_list, so only merger is listed.
        for per_module in (visual.merger, vlm_base.lm_head,
                           vlm_base.model.language_model.embed_tokens):
            per_module.float()

    def build_query_slots(self, cond_slots, cond_valid, seg_slots, seg_valid,
                          B):
        # Both tables are read through nn.Embedding.forward rather than off
        # .weight. For an Embedding left on its default options, forward is a
        # plain row lookup, so asking for every row returns exactly the weight
        # matrix: identical values, identical gradients.
        #
        # Going through forward matters because parameter sharding frameworks
        # hook a module's forward to materialise that module's own parameters;
        # a bare attribute read bypasses the hook and can hand back a
        # placeholder. This keeps the model a plain nn.Module while staying
        # correct under such frameworks.
        device = cond_slots.device
        query_ids = torch.arange(self.query_num, device=device)
        q = self.query_embedding(query_ids)[None].expand(B, -1, -1)

        inject = self.cond_to_query(cond_slots) + self.seg_to_query(seg_slots)
        valid = (cond_valid * seg_valid).unsqueeze(-1).to(inject.dtype)
        null_ids = torch.zeros(1, dtype=torch.long, device=device)
        null = self.null_slot(null_ids)[None].to(inject.dtype)
        inject = inject * valid + null * (1.0 - valid)

        # [B, max_prompt_num, C] -> [B, query_num, C]
        inject = inject.repeat_interleave(self.queries_per_slot, dim=1)

        return q + inject

    def predict(self, backbone_out, query_embeds=None):
        B, H, W, C = backbone_out.shape
        img_seq_len = H * W

        # [B, H*W, C]
        img_tokens = backbone_out.reshape(B, img_seq_len, C)

        # [B, query_num, C]
        # Queries are conditioned per prompt by the caller; slot
        # k carries the condition and segment embedding of
        # prompt k.
        if query_embeds is not None:
            q = query_embeds
        else:
            # See build_query_slots: read the table through forward so that
            # parameter sharding frameworks can materialise it.
            query_ids = torch.arange(self.query_num,
                                     device=backbone_out.device)
            q = self.query_embedding(query_ids)[None].expand(B, -1, -1)

        # Concatenate: [B, query_num + H*W, C]
        # Following dense_segmentation's approach of concatenating query and
        # image tokens.
        #
        # The per-prompt conditioning is carried entirely by build_query_slots
        # (cond_slots + seg_slots injected into the matching query slot). The
        # raw <p>...</p> token sequence is deliberately NOT concatenated here:
        # samples in a batch hold a different number of condition tokens, so it
        # would have to be zero padded, and QueryAttention has no key padding
        # mask. Those zero rows are not neutral (LayerNorm maps them to its
        # bias), so they would behave as extra tokens that every query and
        # image token attends to, making a sample's prediction depend on how
        # long the other samples in the batch happen to be.
        x = torch.cat([q, img_tokens], dim=1)
        for block in self.query_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.feat_norm(x)

        # query embeddings: [B, query_num, C]
        q = x[:, :self.query_num, :]

        # image features: [B, H*W, C]
        img_feat = x[:, self.query_num:self.query_num + img_seq_len, :]
        # [B, C, H, W]
        img_feat = img_feat.transpose(1, 2).reshape(img_feat.shape[0], -1,
                                                    self.grid_size,
                                                    self.grid_size)

        # query projection for mask dot product
        q_proj = self.query_proj(q)

        # upscale image features
        for block in self.upscale_blocks:
            if self.use_gradient_checkpoint:
                img_feat = checkpoint(block, img_feat, use_reentrant=False)
            else:
                img_feat = block(img_feat)

        q_proj = q_proj.float()
        img_feat = img_feat.float()

        # [B, query_num, C] x [B, C, H', W'] -> [B, query_num, H', W']
        mask_preds = torch.einsum("bqc, bchw -> bqhw", q_proj, img_feat)

        mask_preds = F.interpolate(mask_preds,
                                   (self.image_size, self.image_size),
                                   mode="bilinear")

        return mask_preds, q

    def forward_vlm(self,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    mm_token_type_ids=None,
                    labels=None,
                    inputs_embeds=None,
                    position_ids=None):
        if inputs_embeds is not None:
            # ---- inputs_embeds mode (VGDSEG) ----
            # Qwen3.5 does NOT have DeepStack, so we simply call the
            # language_model + lm_head without visual_pos_masks or
            # deepstack_visual_embeds.
            # self.vlm is always a PeftModel (LoRA); access the underlying
            # Qwen3_5ForConditionalGeneration via base_model.model.
            # NOTE: Although we bypass the PeftModel wrapper and call
            # vlm_base.model.language_model directly, the LoRA adapters
            # are still active. This is because get_peft_model() has
            # already replaced the internal Linear layers with LoraLinear
            # modules in-place, so calling through any path that reaches
            # those layers will invoke the LoRA computation.
            vlm_base = self.vlm.base_model.model

            lm_outputs = vlm_base.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = lm_outputs.last_hidden_state

            logits = vlm_base.lm_head(hidden_states)

            loss = None
            if labels is not None:
                loss = vlm_base.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=vlm_base.config.text_config.vocab_size,
                )

            # Build an output object compatible with the standard path.
            # The downstream code accesses .loss and .hidden_states[-1].
            vlm_outputs = types.SimpleNamespace(
                loss=loss,
                logits=logits,
                # Wrap in a tuple so that [-1] indexing works the same
                # as the standard CausalLMOutputWithPast.hidden_states.
                hidden_states=(hidden_states, ),
            )

            return vlm_outputs
        else:
            # ---- Standard mode (REFSEG) ----
            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                labels=labels,
                output_hidden_states=True,
            )
            vlm_outputs = self.vlm(**kwargs)

            return vlm_outputs

    def build_vlm_inputs(self, input_ids, pixel_values, image_grid_thw,
                         mm_token_type_ids, attention_mask, vprompt_feats_list,
                         region_token_id):
        # self.vlm is always a PeftModel (LoRA); access the underlying
        # conditional generation model via base_model.model.
        vlm_base = self.vlm.base_model.model

        inputs_embeds = vlm_base.get_input_embeddings()(input_ids)

        # ---- image tokens ----
        if pixel_values is not None:
            vision_output = vlm_base.get_image_features(
                pixel_values.to(inputs_embeds.dtype), image_grid_thw)
            image_embeds = torch.cat(vision_output.pooler_output,
                                     dim=0).to(inputs_embeds.device,
                                               inputs_embeds.dtype)

            image_mask = (input_ids == vlm_base.config.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_embeds)

        # ---- region tokens ----
        # masked_scatter consumes the source in row-major order of the mask.
        # The mask's True positions run (sample, region, token) and the
        # flattened features run in exactly the same order, so region k's
        # tokens land on region k's placeholders.
        if vprompt_feats_list is not None:
            region_feats = torch.cat(
                [f.reshape(-1, f.shape[-1]) for f in vprompt_feats_list],
                dim=0)
            region_mask = (input_ids == region_token_id)
            assert int(region_mask.sum()) == region_feats.shape[0], (
                f'<region> placeholder count {int(region_mask.sum())} does '
                f'not match the number of region feature tokens '
                f'{region_feats.shape[0]}; check that the tokenizer and the '
                f'model agree on region_token_num')
            inputs_embeds = inputs_embeds.masked_scatter(
                region_mask.unsqueeze(-1).expand_as(inputs_embeds),
                region_feats.to(inputs_embeds.dtype))

        # ---- positions ----
        position_ids, _ = vlm_base.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return inputs_embeds, position_ids

    def forward(self,
                images,
                input_ids=None,
                attention_mask=None,
                pixel_values=None,
                image_grid_thw=None,
                mm_token_type_ids=None,
                cond_ids=None,
                seg_ids=None,
                vprompt_masks=None,
                labels=None,
                task_type=None,
                region_token_id=None,
                valid_sizes=None,
                prompt_type_id=None):
        """Forward pass supporting both text prompt and visual prompt
        segmentation tasks.

        Text prompt (REFSEG):
            VLM processes input_ids + pixel_values normally.
            cond_embeds/seg_embeds extracted from VLM output.
            Uses per-sample independent category space.

        Visual prompt (VGDSEG):
            region_encoder turns each prompt mask into region_token_num
            ordered tokens, which are scattered onto the <region>
            placeholders the tokenizer already emitted.
            VLM can then "see" the visual regions and produce semantically
            meaningful hidden states.
            cond_embeds/seg_embeds extracted from VLM output (same as REFSEG).
            Uses per-sample independent category space.

        Args:
            images: [B, 3, H, W] backbone input images (1024x1024)
            input_ids: [B, L] tokenized text (required for both tasks)
            attention_mask: [B, L] attention mask for VLM
            pixel_values: VLM image pixels (from VLM image processor)
            image_grid_thw: VLM image grid info
            mm_token_type_ids: [B, L] multimodal token type ids from processor
            cond_ids: [B, L] condition IDs marking <p>...</p> token ranges
            seg_ids: [B, L] segment IDs marking <SEG> token positions
            vprompt_masks: list of [N_i, H, W] visual prompt masks
                (only for VGDSEG)
            labels: [B, L] labels for VLM autoregressive loss, or None
            task_type: str or None, "refseg" or "vgdseg". If None, auto-detect
                from vprompt_masks presence.
            region_token_id: int or None, token id for <region>. Required for
                VGDSEG to locate the region placeholders in input_ids.
            valid_sizes: [B, 2] non-padded (height, width) of each image, used
                to normalise the region geometry encoding.
            prompt_type_id: [B] visual prompt type ids (0 point, 1 box,
                2 mask), or None.

        Returns:
            mask_preds: [B, query_num, H, W]
            class_preds: [B, query_num, num_classes]
            vlm_loss: scalar tensor or None (VLM autoregressive loss)
        """
        # Called through __call__ (not .forward()) so that frameworks which
        # hook forward to materialise a module's own parameters -- e.g. ZeRO-3
        # parameter partitioning -- see this call. Invoking .forward() directly
        # bypasses those hooks, leaving ViTImageEncoder's own pos_embed as an
        # unmaterialised placeholder.
        backbone_out = self.backbone(images)

        B = images.shape[0]
        cond_slots = None
        cond_valid = None
        seg_slots = None
        seg_valid = None
        vlm_loss = None

        # Auto-detect task_type if not provided
        is_vgdseg = vprompt_masks is not None
        if task_type is None:
            task_type = "vgdseg" if is_vgdseg else "refseg"

        # ---- Visual prompt: extract region features from backbone ----
        vprompt_feats_list = None
        # An exactly-zero scalar produced by the region encoder on text prompt
        # batches. See the else branch below for why it has to exist.
        region_null_term = None
        if vprompt_masks is not None:
            # [B,C,H,W]
            backbone_feat = backbone_out.permute(0, 3, 1, 2)
            vprompt_feats_list = self.region_encoder(
                backbone_feat,
                vprompt_masks,
                valid_sizes=valid_sizes,
                prompt_type_id=prompt_type_id)
        else:
            # Text prompt batch: the region encoder produces nothing useful, but
            # it must still take part in the backward pass.
            #
            # ZeRO-3 keeps one persistent gradient buffer per parameter shard
            # (stage3.py: grad_partitions_flat_buffer). That buffer is only ever
            # written by the reduce hook of a parameter that actually received a
            # gradient -- partition_grads() does copy_ on micro step 0 and add_
            # afterwards. zero_grad() only sets param.grad = None and never
            # clears the buffer, while the backward epilogue unconditionally
            # rebuilds averaged_gradients for every parameter with
            # requires_grad=True. A submodule skipped on this step therefore
            # gets the gradient of the PREVIOUS step applied a second time.
            # ZeRO-1/2 zero-fill instead (stage_1_and_2.py: get_flat_partition),
            # so only ZeRO-3 is exposed to this.
            #
            # Running the encoder on an all-zero mask and scaling its output by
            # 0.0 keeps the forward result bit-for-bit identical while
            # guaranteeing the hook fires and writes a mathematically correct
            # zero. It also stops the region encoder from being an unused
            # parameter for DDP.
            #
            # vprompt_feats_list stays None, so build_vlm_inputs() is not given
            # any region features and the REFSEG path is untouched.
            dummy_mask = images.new_zeros(1, images.shape[-2],
                                          images.shape[-1])
            backbone_feat = backbone_out.permute(0, 3, 1, 2)
            dummy_feats_list = self.region_encoder(
                backbone_feat, [dummy_mask] * B,
                valid_sizes=valid_sizes,
                prompt_type_id=prompt_type_id)
            region_null_term = sum(per_feats.sum()
                                   for per_feats in dummy_feats_list) * 0.0

        # ---- Run VLM forward ----
        if input_ids is not None:
            # Every VLM call has to run inside an autocast region.
            #
            # The VLM is loaded in bf16 and only its trainable modules are cast
            # to fp32 (see __init__), so its frozen and its trainable halves
            # never share a dtype: the vision tower always runs in bf16
            # (transformers casts pixel_values to visual.dtype internally) and
            # hands bf16 activations to the fp32 merger. F.layer_norm does not
            # promote its inputs, so a plain fp32 forward raises "expected
            # scalar type BFloat16 but found Float", while autocast unifies
            # input and weight for every op it covers.
            #
            # When the caller already opened an amp region its dtype is reused,
            # so the VLM follows the precision the run was configured with.
            # Otherwise bf16 -- the dtype the frozen weights are stored in --
            # is entered explicitly.
            device_type = backbone_out.device.type
            if torch.is_autocast_enabled(device_type):
                amp_type = torch.get_autocast_dtype(device_type)
            else:
                amp_type = torch.bfloat16

            with torch.autocast(device_type=device_type, dtype=amp_type):
                if task_type == "vgdseg" and vprompt_feats_list is not None \
                        and region_token_id is not None:
                    # VGDSEG: scatter region features onto the <region>
                    # placeholders, then forward with inputs_embeds. The
                    # sequence length is unchanged, so attention_mask / labels
                    # / cond_ids / seg_ids / mm_token_type_ids all stay valid
                    # as-is.
                    inputs_embeds, position_ids = self.build_vlm_inputs(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        mm_token_type_ids=mm_token_type_ids,
                        attention_mask=attention_mask,
                        vprompt_feats_list=vprompt_feats_list,
                        region_token_id=region_token_id,
                    )
                    vlm_outputs = self.forward_vlm(
                        input_ids=None,
                        attention_mask=attention_mask,
                        pixel_values=None,
                        image_grid_thw=image_grid_thw,
                        mm_token_type_ids=mm_token_type_ids,
                        labels=labels,
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                    )
                else:
                    # REFSEG: standard VLM forward with input_ids
                    vlm_outputs = self.forward_vlm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        mm_token_type_ids=mm_token_type_ids,
                        labels=labels,
                    )

            # Extract VLM loss (when labels provided)
            vlm_loss = vlm_outputs.loss

            # Get last hidden state
            vlm_hidden = vlm_outputs.hidden_states[-1]

            # Project to decoder space
            vlm_hidden = vlm_hidden.to(backbone_out.dtype)
            mlm_embeds = self.mlm_projector(vlm_hidden)

            # REFSEG and VGDSEG are handled identically here: cond_ids mark
            # <p>...</p> spans (a phrase for REFSEG, a group of region
            # placeholders for VGDSEG) and seg_ids mark one <SEG> per prompt.
            if cond_ids is not None:
                cond_slots, cond_valid = self.cond_readout(
                    mlm_embeds, cond_ids, self.max_prompt_num)
            if seg_ids is not None:
                seg_slots, seg_valid = self.seg_readout(
                    mlm_embeds, seg_ids, self.max_prompt_num)

        # ---- decoder forward ----
        # Query slot k is conditioned on prompt k.
        if cond_slots is not None and seg_slots is not None:
            query_embeds = self.build_query_slots(cond_slots, cond_valid,
                                                  seg_slots, seg_valid, B)
        else:
            query_embeds = None

        mask_preds, q_out = self.predict(backbone_out,
                                         query_embeds=query_embeds)

        # Attach the zero-valued region encoder term built in the text prompt
        # branch above. The value is exactly 0.0, so mask_preds is numerically
        # unchanged; only the backward graph gains an edge to the region encoder.
        if region_null_term is not None:
            mask_preds = mask_preds + region_null_term.to(mask_preds.dtype)

        # ---- class prediction with condition embeddings ----
        # The class space is [prompt_0 ... prompt_{n-1}, background] where n
        # is the largest number of prompts actually present in this batch.
        # Keeping it dynamic (rather than always max_prompt_num) preserves
        # the existing loss semantics, in which the background class index is
        # class_preds.shape[-1] - 1 and class_gts holds the prompt index.
        # See build_query_slots: read the table through forward so that
        # parameter sharding frameworks can materialise it.
        bg_ids = torch.zeros(1, dtype=torch.long, device=q_out.device)
        bg = self.bg_embeds(bg_ids)[None].expand(B, -1, -1)
        if cond_slots is not None:
            num_classes = max(int(cond_valid.sum(dim=1).max().item()), 1)
            cond_embeds = cond_slots[:, :num_classes]
            embed_masks = cond_valid[:, :num_classes]

            cond_with_bg = torch.cat([cond_embeds, bg], dim=1)
            bg_mask = torch.ones(B,
                                 1,
                                 device=embed_masks.device,
                                 dtype=embed_masks.dtype)
            embed_masks_with_bg = torch.cat([embed_masks, bg_mask], dim=1)
            class_preds = self.class_pred(q_out, cond_with_bg,
                                          embed_masks_with_bg)
        else:
            class_preds = self.class_pred(q_out, bg, None)

        return mask_preds, class_preds, vlm_loss


def qwen35_sam_vit_base_patch16_prompt_segmentation(image_size=1024,
                                                    patch_size=16,
                                                    **kwargs):
    return QWEN35SAMPromptSegmentation(image_size=image_size,
                                       patch_size=patch_size,
                                       embedding_planes=768,
                                       block_nums=12,
                                       head_nums=12,
                                       mlp_ratio=4,
                                       window_size=14,
                                       global_attn_indexes=[2, 5, 8, 11],
                                       **kwargs)


def qwen35_sam_vit_large_patch16_prompt_segmentation(image_size=1024,
                                                     patch_size=16,
                                                     **kwargs):
    return QWEN35SAMPromptSegmentation(image_size=image_size,
                                       patch_size=patch_size,
                                       embedding_planes=1024,
                                       block_nums=24,
                                       head_nums=16,
                                       mlp_ratio=4,
                                       window_size=14,
                                       global_attn_indexes=[5, 11, 17, 23],
                                       **kwargs)


def qwen35_sam_vit_huge_patch16_prompt_segmentation(image_size=1024,
                                                    patch_size=16,
                                                    **kwargs):
    return QWEN35SAMPromptSegmentation(image_size=image_size,
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

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, GenerationConfig
    from PIL import Image

    qwen35_model_path = "Qwen/Qwen3.5-4B"
    qwen35_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen35_model_path,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2')
    qwen35_model = qwen35_model.cuda()
    qwen35_processor = AutoProcessor.from_pretrained(qwen35_model_path)

    test_image_path = "/root/code/SimplePromptSegmentation_pytorch_training_examples/SimplerPromptSegmentation/models/demo.jpg"
    test_image = Image.open(test_image_path)

    # Construct messages: image + text prompt
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image",
                "image": test_image,
            },
            {
                "type": "text",
                "text": "Describe this image.",
            },
        ],
    }]

    # thinking mode inference test case
    # Use processor to convert messages into model inputs (with enable_thinking=True)
    inputs = qwen35_processor.apply_chat_template(messages,
                                                  tokenize=True,
                                                  add_generation_prompt=True,
                                                  enable_thinking=True,
                                                  return_dict=True,
                                                  return_tensors="pt")
    inputs = inputs.to(qwen35_model.device)

    print(f"input keys: {list(inputs.keys())}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"mm_token_type_ids shape: {inputs['mm_token_type_ids'].shape}")
    print(f"pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"image_grid_thw: {inputs['image_grid_thw']}")

    # Generate response
    generated_ids = qwen35_model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen35_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    print(f"Qwen3.5 thinking mode output: {output_text}")

    # Non-thinking mode inference test case
    # Use processor to convert messages into model inputs (with enable_thinking=False)
    inputs = qwen35_processor.apply_chat_template(messages,
                                                  tokenize=True,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False,
                                                  return_dict=True,
                                                  return_tensors="pt")
    inputs = inputs.to(qwen35_model.device)

    print(f"input keys: {list(inputs.keys())}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"mm_token_type_ids shape: {inputs['mm_token_type_ids'].shape}")
    print(f"pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"image_grid_thw: {inputs['image_grid_thw']}")

    # Generate response
    generated_ids = qwen35_model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen35_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    print(f"Qwen3.5 non-thinking mode output: {output_text}")

    del qwen35_model
    torch.cuda.empty_cache()
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    import json
    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    visual_image_path = '/root/code/SimplePromptSegmentation_pytorch_training_examples/SimplerPromptSegmentation/models/sa_1.jpg'
    visual_json_path = '/root/code/SimplePromptSegmentation_pytorch_training_examples/SimplerPromptSegmentation/models/sa_1.json'

    visual_image = cv2.imdecode(np.fromfile(visual_image_path, dtype=np.uint8),
                                cv2.IMREAD_COLOR)
    visual_image = cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB)
    visual_image = visual_image.astype(np.float32)
    visual_pil_image = Image.open(visual_image_path).convert('RGB')

    with open(visual_json_path, encoding='utf-8') as f:
        visual_json_data = json.load(f)
    visual_annotations = visual_json_data['annotations']
    first_annot = visual_annotations[0]
    visual_gt_mask = mask_utils.decode(first_annot['segmentation'])
    visual_gt_mask[visual_gt_mask > 0] = 1
    visual_gt_mask = visual_gt_mask.astype(np.float32)
    print(f"Visual prompt GT mask shape: {visual_gt_mask.shape}")

    image_size = 1024
    h_origin, w_origin = visual_image.shape[:2]
    factor = image_size / max(h_origin, w_origin)
    resize_h, resize_w = int(round(h_origin * factor)), int(
        round(w_origin * factor))
    resized_visual_image = cv2.resize(visual_image, (resize_w, resize_h))
    resized_visual_pil_image = visual_pil_image.resize((resize_w, resize_h))
    resized_gt_mask = cv2.resize(visual_gt_mask, (resize_w, resize_h),
                                 interpolation=cv2.INTER_NEAREST)

    # Use GT mask as visual prompt mask (vprompt_mask), pad to 1024x1024
    vprompt_mask_padded = np.zeros((image_size, image_size), dtype=np.float32)
    vprompt_mask_padded[:resize_h, :resize_w] = resized_gt_mask

    # Normalize and pad image to 1024x1024
    MEAN = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    STD = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
    normed_image = (resized_visual_image - MEAN) / STD
    input_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    input_image[:resize_h, :resize_w, :] = normed_image

    # Convert to tensor [1, 3, 1024, 1024]
    input_image_tensor = torch.from_numpy(input_image).permute(
        2, 0, 1).unsqueeze(0).float()

    # vprompt_masks: list of [N_i, H, W] tensors, one per batch element
    # [1, 1024, 1024]
    vprompt_masks_tensor = [torch.from_numpy(vprompt_mask_padded).unsqueeze(0)]

    from tokenizer import Qwen35SegTokenizer
    seg_tokenizer = Qwen35SegTokenizer(qwen35_model_path)

    # ---- Build VLM inputs using tokenizer ----
    visual_prompt_texts = ["<region>"]
    visual_sample_type = "visual"
    visual_prompt_languages = ["english"]

    # Step 1: build_conversations —— 将原始 prompt 文本转换为 question/answer 文本
    visual_question_texts, visual_answer_texts, _ = seg_tokenizer.build_conversations(
        prompt_texts=visual_prompt_texts,
        sample_type=visual_sample_type,
        prompt_languages=visual_prompt_languages,
    )
    print(f"visual_question_texts: {visual_question_texts}")
    print(f"visual_answer_texts: {visual_answer_texts}")
    # Step 2: build_chat_messages —— 将 question/answer 文本 + 图片组织成 Qwen3.5 聊天消息格式
    visual_messages = seg_tokenizer.build_chat_messages(
        question_text=visual_question_texts[0],
        answer_text=visual_answer_texts[0],
        pil_image=resized_visual_pil_image,
    )
    print(f"visual_messages: {visual_messages}")

    # Step 3: apply_chat_template（tokenize=False）—— 获取格式化后的纯文本，不做 tokenization
    visual_inputs_text = seg_tokenizer.processor.apply_chat_template(
        visual_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print(f"visual_inputs: {visual_inputs_text}")

    visual_tokenized = seg_tokenizer.encode(
        prompt_texts=visual_prompt_texts,
        sample_type=visual_sample_type,
        prompt_languages=visual_prompt_languages,
        pil_images=[resized_visual_pil_image],
    )
    print(f"visual prompt input keys: {list(visual_tokenized.keys())}")
    print(
        f"visual prompt input_ids shape: {visual_tokenized['input_ids'].shape}"
    )
    print(
        f"visual prompt attention_mask shape: {visual_tokenized['attention_mask'].shape}"
    )
    print(
        f"visual prompt mm_token_type_ids shape: {visual_tokenized['mm_token_type_ids'].shape}"
    )
    print(
        f"visual prompt pixel_values shape: {visual_tokenized['pixel_values'].shape}"
    )
    print(
        f"visual prompt image_grid_thw: {visual_tokenized['image_grid_thw']}")

    model = qwen35_sam_vit_base_patch16_prompt_segmentation(
        vlm_model_path=qwen35_model_path,
        tokenizer_vocab_size=seg_tokenizer.vocab_size)
    model = model.cuda()
    model.eval()

    input_image_tensor = input_image_tensor.cuda()
    vprompt_masks_device = [m.cuda() for m in vprompt_masks_tensor]
    vlm_input_ids = visual_tokenized['input_ids'].cuda()
    vlm_attention_mask = visual_tokenized['attention_mask'].cuda()
    vlm_pixel_values = visual_tokenized['pixel_values'].cuda()
    vlm_image_grid_thw = visual_tokenized['image_grid_thw'].cuda()
    vlm_mm_token_type_ids = visual_tokenized['mm_token_type_ids'].cuda()
    vlm_cond_ids = visual_tokenized['cond_ids'].cuda()
    vlm_seg_ids = visual_tokenized['seg_ids'].cuda()
    vlm_labels = visual_tokenized['labels'].cuda()
    # region_token_id is passed so build_vlm_inputs can replace the
    # <region> placeholder embeddings with region_encoder features
    # via a single masked_scatter.

    with torch.no_grad():
        visual_mask_preds, visual_class_preds, visual_vlm_loss = model(
            images=input_image_tensor,
            input_ids=vlm_input_ids,
            attention_mask=vlm_attention_mask,
            pixel_values=vlm_pixel_values,
            image_grid_thw=vlm_image_grid_thw,
            mm_token_type_ids=vlm_mm_token_type_ids,
            cond_ids=vlm_cond_ids,
            seg_ids=vlm_seg_ids,
            vprompt_masks=vprompt_masks_device,
            labels=vlm_labels,
            region_token_id=seg_tokenizer.region_token_id,
        )

    print(f"visual prompt mask_preds shape: {visual_mask_preds.shape}")
    print(f"visual prompt class_preds shape: {visual_class_preds.shape}")
    print(f"visual prompt vlm_loss: {visual_vlm_loss}")

    del model
    torch.cuda.empty_cache()
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    text_image_path = '/root/code/SimplePromptSegmentation_pytorch_training_examples/SimplerPromptSegmentation/models/sa_100000.jpg'
    text_json_path = '/root/code/SimplePromptSegmentation_pytorch_training_examples/SimplerPromptSegmentation/models/sa_100000.json'

    text_image = cv2.imdecode(np.fromfile(text_image_path, dtype=np.uint8),
                              cv2.IMREAD_COLOR)
    text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2RGB)
    text_image = text_image.astype(np.float32)
    text_pil_image = Image.open(text_image_path).convert('RGB')

    with open(text_json_path, encoding='utf-8') as f:
        text_json_data = json.load(f)
    first_mask_name = list(text_json_data.keys())[0]
    first_mask_info = text_json_data[first_mask_name]
    text_description = first_mask_info['english'][
        'absolute_detail_description']
    # [x_min, y_min, w, h]
    mask_box = first_mask_info['mask_box']
    print(f"Text prompt description: {text_description}")
    print(f"Text prompt mask_box: {mask_box}")

    image_size = 1024
    h_origin_t, w_origin_t = text_image.shape[:2]
    factor_t = image_size / max(h_origin_t, w_origin_t)
    resize_h_t = int(round(h_origin_t * factor_t))
    resize_w_t = int(round(w_origin_t * factor_t))

    resized_text_image = cv2.resize(text_image, (resize_w_t, resize_h_t))
    resized_text_pil_image = text_pil_image.resize((resize_w_t, resize_h_t))

    # Normalize and pad image to 1024x1024
    MEAN = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    STD = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
    normed_text_image = (resized_text_image - MEAN) / STD
    input_text_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    input_text_image[:resize_h_t, :resize_w_t, :] = normed_text_image

    # Convert to tensor [1, 3, 1024, 1024]
    input_text_image_tensor = torch.from_numpy(input_text_image).permute(
        2, 0, 1).unsqueeze(0).float()

    from tokenizer import Qwen35SegTokenizer
    seg_tokenizer = Qwen35SegTokenizer(qwen35_model_path)

    # ---- Build VLM inputs using tokenizer ----
    text_prompt_texts = [[text_description]]
    text_sample_type = "text"
    text_prompt_languages = ["english"]

    # Step 1: build_conversations —— 将原始 prompt 文本转换为 question/answer 文本
    text_question_texts, text_answer_texts, _ = seg_tokenizer.build_conversations(
        prompt_texts=text_prompt_texts,
        sample_type=text_sample_type,
        prompt_languages=text_prompt_languages,
    )
    print(f"text_question_texts: {text_question_texts}")
    print(f"text_answer_texts: {text_answer_texts}")

    # Step 2: build_chat_messages —— 将 question/answer 文本 + 图片组织成 Qwen3.5 聊天消息格式
    text_messages = seg_tokenizer.build_chat_messages(
        question_text=text_question_texts[0],
        answer_text=text_answer_texts[0],
        pil_image=resized_text_pil_image,
    )
    print(f"text_messages: {text_messages}")

    # Step 3: apply_chat_template（tokenize=False）—— 获取格式化后的纯文本，不做 tokenization
    text_inputs_text = seg_tokenizer.processor.apply_chat_template(
        text_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print(f"text_inputs: {text_inputs_text}")

    text_tokenized = seg_tokenizer.encode(
        prompt_texts=text_prompt_texts,
        sample_type=text_sample_type,
        prompt_languages=text_prompt_languages,
        pil_images=[resized_text_pil_image],
    )
    print(f"text prompt input keys: {list(text_tokenized.keys())}")
    print(f"text prompt input_ids shape: {text_tokenized['input_ids'].shape}")
    print(
        f"text prompt attention_mask shape: {text_tokenized['attention_mask'].shape}"
    )
    print(
        f"text prompt mm_token_type_ids shape: {text_tokenized['mm_token_type_ids'].shape}"
    )
    print(
        f"text prompt pixel_values shape: {text_tokenized['pixel_values'].shape}"
    )
    print(f"text prompt image_grid_thw: {text_tokenized['image_grid_thw']}")

    model = qwen35_sam_vit_base_patch16_prompt_segmentation(
        vlm_model_path=qwen35_model_path,
        tokenizer_vocab_size=seg_tokenizer.vocab_size)
    model = model.cuda()
    model.eval()

    # Move all inputs to device
    input_text_image_tensor = input_text_image_tensor.cuda()

    text_vlm_input_ids = text_tokenized['input_ids'].cuda()
    text_vlm_attention_mask = text_tokenized['attention_mask'].cuda()
    text_vlm_pixel_values = text_tokenized['pixel_values'].cuda()
    text_vlm_image_grid_thw = text_tokenized['image_grid_thw'].cuda()
    text_vlm_mm_token_type_ids = text_tokenized['mm_token_type_ids'].cuda()
    text_vlm_cond_ids = text_tokenized['cond_ids'].cuda()
    text_vlm_seg_ids = text_tokenized['seg_ids'].cuda()
    text_vlm_labels = text_tokenized['labels'].cuda()

    with torch.no_grad():
        text_mask_preds, text_class_preds, text_vlm_loss = model(
            images=input_text_image_tensor,
            input_ids=text_vlm_input_ids,
            attention_mask=text_vlm_attention_mask,
            pixel_values=text_vlm_pixel_values,
            image_grid_thw=text_vlm_image_grid_thw,
            mm_token_type_ids=text_vlm_mm_token_type_ids,
            cond_ids=text_vlm_cond_ids,
            seg_ids=text_vlm_seg_ids,
            labels=text_vlm_labels,
        )

    print(f"text prompt mask_preds shape: {text_mask_preds.shape}")
    print(f"text prompt class_preds shape: {text_class_preds.shape}")
    print(f"text prompt vlm_loss: {text_vlm_loss}")

    del model
    torch.cuda.empty_cache()
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
