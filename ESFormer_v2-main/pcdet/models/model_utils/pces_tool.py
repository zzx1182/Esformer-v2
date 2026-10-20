# pcdet/models/model_utils/pces_tool.py


from ...utils import common_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from torch.nn.init import trunc_normal_


class PillarAttentionSampling(nn.Module):
    """
    Pillar-Attention采样网络
    用于从每个Pillar内选择最重要的点
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_heads=8, max_points=128):
        super().__init__()
        self.max_points = max_points
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 线性变换层用于生成Q, K, V
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)

        # 输出权重预测层
        self.weight_proj = nn.Linear(hidden_dim, 1)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(max_points, input_dim))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, pillar_points, pillar_masks):
        """
        Args:
            pillar_points: [B, H, W, max_points, 4] - 每个pillar的点云
            pillar_masks: [B, H, W, max_points] - 有效点的掩码
        Returns:
            selected_points: [B, H, W, N, 4] - 筛选后的点
            selection_weights: [B, H, W, N] - 选择权重
        """
        B, H, W, P, C = pillar_points.shape

        # 添加位置编码
        points_with_pos = pillar_points + self.pos_encoding[:P].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 计算Q, K, V
        Q = self.q_proj(points_with_pos)  # [B, H, W, P, hidden_dim]
        K = self.k_proj(points_with_pos)
        V = self.v_proj(points_with_pos)

        # Multi-head attention
        Q = Q.view(B, H, W, P, self.num_heads, self.head_dim).transpose(-2, -1)
        K = K.view(B, H, W, P, self.num_heads, self.head_dim).transpose(-2, -1)
        V = V.view(B, H, W, P, self.num_heads, self.head_dim).transpose(-2, -1)

        # 计算注意力分数 (移除Softmax)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码
        mask_expanded = pillar_masks.unsqueeze(-2).unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(~mask_expanded, -1e9)

        # 聚合多头结果
        attention_output = torch.matmul(attention_scores, V)
        attention_output = attention_output.transpose(-2, -1).contiguous()
        attention_output = attention_output.view(B, H, W, P, -1)

        # 计算选择权重α (不使用Softmax)
        weights = self.weight_proj(attention_output).squeeze(-1)  # [B, H, W, P]
        weights = torch.sigmoid(weights)  # 使用sigmoid替代softmax

        # 应用原始掩码
        weights = weights * pillar_masks.float()

        # 选择权重大于0.7的点
        threshold = 0.7
        selection_mask = (weights > threshold) & pillar_masks

        # 为了保持固定输出大小，我们选择top-k个点
        max_selected = min(64, P)  # 每个pillar最多选择64个点

        selected_points_list = []
        selection_weights_list = []

        for b in range(B):
            for h in range(H):
                for w in range(W):
                    valid_mask = selection_mask[b, h, w]
                    if valid_mask.sum() > 0:
                        valid_indices = torch.where(valid_mask)[0]
                        valid_weights = weights[b, h, w, valid_indices]

                        # 选择top-k
                        if len(valid_indices) > max_selected:
                            top_k_indices = torch.topk(valid_weights, max_selected)[1]
                            selected_indices = valid_indices[top_k_indices]
                        else:
                            selected_indices = valid_indices

                        selected_pts = pillar_points[b, h, w, selected_indices]
                        selected_wts = weights[b, h, w, selected_indices]
                    else:
                        # 如果没有选中的点，选择第一个有效点
                        first_valid = torch.where(pillar_masks[b, h, w])[0]
                        if len(first_valid) > 0:
                            selected_pts = pillar_points[b, h, w, first_valid[:1]]
                            selected_wts = weights[b, h, w, first_valid[:1]]
                        else:
                            selected_pts = torch.zeros(1, C, device=pillar_points.device)
                            selected_wts = torch.zeros(1, device=pillar_points.device)

                    # 填充到固定大小
                    if selected_pts.shape[0] < max_selected:
                        pad_size = max_selected - selected_pts.shape[0]
                        pad_pts = torch.zeros(pad_size, C, device=pillar_points.device)
                        pad_wts = torch.zeros(pad_size, device=pillar_points.device)
                        selected_pts = torch.cat([selected_pts, pad_pts], dim=0)
                        selected_wts = torch.cat([selected_wts, pad_wts], dim=0)

                    selected_points_list.append(selected_pts)
                    selection_weights_list.append(selected_wts)

        # 重新组织输出
        selected_points = torch.stack(selected_points_list).view(B, H, W, max_selected, C)
        selection_weights = torch.stack(selection_weights_list).view(B, H, W, max_selected)

        return selected_points, selection_weights


class PointCloudExpansion(nn.Module):
    """
    点云扩展单元
    将筛选出的关键点扩散到相邻的Pillar中
    """

    def __init__(self, max_points=128):
        super().__init__()
        self.max_points = max_points

        # 8个相邻方向的偏移
        self.neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

    def farthest_point_sampling(self, points, num_samples):
        """
        最远点采样算法
        """
        N, C = points.shape
        if N <= num_samples:
            return points

        device = points.device
        centroids = torch.zeros(num_samples, dtype=torch.long, device=device)
        distance = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), device=device)

        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :3]  # 只使用xyz坐标计算距离
            dist = torch.sum((points[:, :3] - centroid) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, 0)[1]

        return points[centroids]

    def forward(self, selected_points, selection_weights, original_pillar_points):
        """
        Args:
            selected_points: [B, H, W, N, 4] - Pillar-Attention输出的点
            selection_weights: [B, H, W, N] - 选择权重
            original_pillar_points: [B, H, W, max_points, 4] - 原始pillar点云
        Returns:
            expanded_points: [B, H, W, max_points, 4] - 扩展后的点云
        """
        B, H, W, N, C = selected_points.shape
        expanded_points = original_pillar_points.clone()

        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # 获取当前pillar的选中点
                    current_points = selected_points[b, h, w]
                    current_weights = selection_weights[b, h, w]

                    # 过滤出有效点
                    valid_mask = current_weights > 0
                    if not valid_mask.any():
                        continue

                    valid_points = current_points[valid_mask]

                    # 将这些点扩散到8个相邻的pillar中
                    for dh, dw in self.neighbor_offsets:
                        nh, nw = h + dh, w + dw

                        # 检查边界
                        if 0 <= nh < H and 0 <= nw < W:
                            # 获取邻居pillar的现有点
                            neighbor_points = expanded_points[b, nh, nw]
                            neighbor_mask = torch.sum(neighbor_points, dim=1) != 0
                            existing_points = neighbor_points[neighbor_mask]

                            # 合并当前点和邻居点
                            if len(existing_points) > 0:
                                combined_points = torch.cat([existing_points, valid_points], dim=0)
                            else:
                                combined_points = valid_points

                            # 使用FPS重采样
                            if len(combined_points) > self.max_points:
                                sampled_points = self.farthest_point_sampling(
                                    combined_points, self.max_points
                                )
                            else:
                                sampled_points = combined_points

                            # 更新邻居pillar
                            expanded_points[b, nh, nw].zero_()
                            expanded_points[b, nh, nw, :len(sampled_points)] = sampled_points

        return expanded_points


class PCESTool(nn.Module):
    """
    完整的 PCES-Tool (点云扩展采样工具) 模块。
    它将被主检测器类 (ESFormer) 在初始化时构建。
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # 从 YAML 配置文件中读取参数
        hidden_dim = self.model_cfg.HIDDEN_DIM
        num_heads = self.model_cfg.NUM_HEADS
        max_points_per_pillar = self.model_cfg.MAX_POINTS_PER_PILLAR

        self.pillar_attention = PillarAttentionSampling(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_points=max_points_per_pillar
        )
        self.point_expansion = PointCloudExpansion(max_points=max_points_per_pillar)

    def forward(self, batch_dict):
        """
        前向传播函数。
        Args:
            batch_dict (dict): 包含以下键值对
                - pillar_points: (B, H, W, P, C) -> 每个 Pillar 包含的点
                - pillar_masks: (B, H, W, P) -> 对应的有效点掩码
        Returns:
            batch_dict (dict): 添加了以下新键值对
                - enhanced_points: (B, H, W, P, C) -> 经过扩展和采样后的点
                - enhanced_masks: (B, H, W, P) -> 对应的新掩码
        """
        pillar_points = batch_dict['pillar_points']
        pillar_masks = batch_dict['pillar_masks']

        # 步骤 1: Pillar-Attention 采样
        selected_points, selection_weights = self.pillar_attention(pillar_points, pillar_masks)

        # 步骤 2: 点云扩展
        expanded_points = self.point_expansion(selected_points, selection_weights, pillar_points)

        # 为扩展后的点云生成新的掩码
        # 如果一个点的所有特征加起来不为0，则我们认为它是一个有效的点
        expanded_masks = torch.sum(expanded_points, dim=-1) != 0

        # 将结果存回 batch_dict 中，供后续模块使用
        batch_dict['enhanced_points'] = expanded_points
        batch_dict['enhanced_masks'] = expanded_masks

        return batch_dict