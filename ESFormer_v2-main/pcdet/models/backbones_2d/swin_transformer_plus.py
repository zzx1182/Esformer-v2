# 文件路径: pcdet/models/backbones_2d/swin_transformer_plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import numpy as np


# ===========================================================================
# 辅助类：Window Attention
# ===========================================================================

class WindowAttention(nn.Module):
    """
    带掩码的窗口多头自注意力机制 (Window-based Multi-head Self Attention)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取相对坐标索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===========================================================================
# Swin Transformer Block
# ===========================================================================

class SwinTransformerBlock(nn.Module):
    """
    带掩码优化的Swin Transformer Block
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 这些将由外部设置
        self.H = None
        self.W = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # 使用简化的 DropPath 实现
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def window_partition(self, x, window_size):
        """将输入分割为窗口"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """将窗口重新组合为特征图"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, mask_matrix=None, sparse_mask=None):
        """
        Args:
            x: [B, H*W, C]
            mask_matrix: window attention mask
            sparse_mask: [B, H*W] - 稀疏掩码，True表示该位置需要计算
        """
        # 确保 H, W 已经设置
        if self.H is None or self.W is None:
            raise ValueError("H and W must be set before forward pass")
            
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 处理稀疏掩码 - 如果大部分区域都是空的，可以跳过计算
        if sparse_mask is not None:
            valid_ratio = sparse_mask.float().mean()
            if valid_ratio < 0.1:  # 如果有效位置少于10%，跳过注意力计算
                x = x.view(B, H * W, C)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x

        # Padding 到窗口大小的倍数
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环移位 (Shifted Window)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割为windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window-based 多头自注意力
        attn_windows = self.attn(x_windows, mask=mask_matrix)

        # 合并windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 移除padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ===========================================================================
# Patch Merging
# ===========================================================================

class PatchMerging(nn.Module):
    """带掩码的Patch Merging层"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W, sparse_mask=None):
        """
        Args:
            x: [B, H*W, C]
            sparse_mask: [B, H*W] - 稀疏掩码
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 2x2 patch merging
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        # 处理稀疏掩码
        new_sparse_mask = None
        if sparse_mask is not None:
            mask_2d = sparse_mask.view(B, H, W)
            new_H, new_W = H // 2, W // 2
            new_mask = torch.zeros(B, new_H, new_W, dtype=torch.bool, device=x.device)
            
            for i in range(new_H):
                for j in range(new_W):
                    region_mask = mask_2d[:, i*2:(i+1)*2, j*2:(j+1)*2]
                    new_mask[:, i, j] = region_mask.any(dim=(1, 2))
            
            new_sparse_mask = new_mask.view(B, -1)

        return x, new_sparse_mask


# ===========================================================================
# Basic Layer
# ===========================================================================

class BasicLayer(nn.Module):
    """ Swin Transformer 的一个基础层，包含多个 Block """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        # 构建 Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, 
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Patch Merging 层 (下采样)
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        """创建用于 SW-MSA 的注意力掩码"""
        if self.window_size <= 0:
            return None
            
        # 创建用于 shifted window attention 的掩码
        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将掩码分割为窗口
        mask_windows = self.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def window_partition(self, x, window_size):
        """将输入分割为窗口"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def forward(self, x, H, W, sparse_mask=None):
        """前向传播"""
        B, L, C = x.shape
        assert L == H * W, f"输入特征尺寸不匹配: L={L}, H*W={H*W}"

        # 为每个 block 创建注意力掩码
        attn_masks = []
        for i, blk in enumerate(self.blocks):
            if blk.shift_size > 0:
                attn_mask = self.create_mask(x, H, W)
            else:
                attn_mask = None
            attn_masks.append(attn_mask)

        # 通过所有的 Transformer blocks
        for i, blk in enumerate(self.blocks):
            blk.H = H
            blk.W = W
            x = blk(x, mask_matrix=attn_masks[i], sparse_mask=sparse_mask)

        # 将特征重塑为 2D 用于输出（多尺度特征）
        x_out = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # 下采样 (如果存在)
        if self.downsample is not None:
            x_down, sparse_mask_down = self.downsample(x, H, W, sparse_mask)
            H_down, W_down = H // 2, W // 2
            return x_out, H, W, x_down, H_down, W_down, sparse_mask_down
        else:
            return x_out, H, W, x, H, W, sparse_mask


# ===========================================================================
# 主模块：SwinTransformerPlus (适配 OpenPCDet)
# ===========================================================================

class SwinTransformerPlus(nn.Module):
    """
    Swin-T+: 适配 OpenPCDet 的最终版本
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        # 从 YAML 配置文件中加载参数
        patch_size = self.model_cfg.get('PATCH_SIZE', [4, 4])
        embed_dim = self.model_cfg.EMBED_DIM
        depths = self.model_cfg.DEPTHS
        num_heads = self.model_cfg.NUM_HEADS
        window_size = self.model_cfg.WINDOW_SIZE
        mlp_ratio = self.model_cfg.get('MLP_RATIO', 4.)
        qkv_bias = self.model_cfg.get('QKV_BIAS', True)
        drop_rate = self.model_cfg.get('DROP_RATE', 0.)
        attn_drop_rate = self.model_cfg.get('ATTN_DROP_RATE', 0.)
        drop_path_rate = self.model_cfg.get('DROP_PATH_RATE', 0.1)
        norm_layer = nn.LayerNorm

        # 将输入 BEV 图转换为 Patch Embedding
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 构建所有 Swin Transformer 层
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(depths) - 1) else None,
            )
            self.layers.append(layer)

        # 计算并设置 OpenPCDet 需要的输出特征维度
        self.num_bev_features = int(embed_dim * 2 ** (len(depths) - 1))

    def forward(self, batch_dict):
        """
        前向传播函数
        Args:
            batch_dict (dict): 包含 'spatial_features': (B, C, H, W)
        Returns:
            batch_dict (dict): 添加了 'spatial_features_2d'
        """
        x = batch_dict['spatial_features']  # 输入的 BEV 特征图

        # Patch embedding
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x = self.pos_drop(x)

        # 逐层处理
        sparse_mask = None
        for layer in self.layers:
            x_out, H_cur, W_cur, x, H_new, W_new, sparse_mask = layer(x, H, W, sparse_mask)
            # 更新当前的 H, W 为下一层做准备
            H, W = H_new, W_new

        # 更新 batch_dict，使用最后一层的输出
        batch_dict['spatial_features_2d'] = x_out
        
        return batch_dict