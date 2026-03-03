# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ==================== VGGT 主模型定义 ====================
# VGGT (Visual Geometry Grounded Transformer) 是 Meta 开发的视觉几何基础模型
# 它从多帧 RGB 图像直接预测深度、位姿、3D点和2D跟踪
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator       # 核心：交替注意力聚合器（DINOv2 骨干网络）
from vggt.heads.camera_head import CameraHead        # 相机位姿预测头（输出 9 维位姿编码）
from vggt.heads.dpt_head import DPTHead              # DPT 风格密集预测头（深度/3D点）
from vggt.heads.track_head import TrackHead          # 2D 点跟踪头


class VGGT(nn.Module, PyTorchModelHubMixin):
    """
    VGGT (Visual Geometry Grounded Transformer) 主模型。
    
    架构概览：
    1. Aggregator（交替注意力聚合器）：基于 DINOv2-Large 的双流交替注意力
       - frame_blocks: 帧内注意力（每帧独立处理，学习帧内特征）
       - global_blocks: 全局注意力（所有帧联合处理，学习帧间几何关系）
       - 输出: 2*embed_dim 维拼接特征 + target_tokens（Layer 22 的注意力用于回环验证）
    
    2. 四个预测头（从 Aggregator 的拼接特征中解码）：
       - CameraHead: 预测 9 维位姿编码 (3T + 4Q + 2FoV)，对应论文 §III
       - DPTHead (depth): 预测深度图 (H,W,1) + 置信度 (H,W)
       - DPTHead (point): 预测3D点 (H,W,3) + 置信度（VGGT-SLAM 中未使用）
       - TrackHead: 预测2D点跟踪（VGGT-SLAM 中未使用）
    
    参数说明：
        img_size: 输入图像大小（默认 518，需为 patch_size 的倍数）
        patch_size: ViT patch 大小（默认 14，来自 DINOv2）
        embed_dim: token 嵌入维度（默认 1024，DINOv2-Large）
        target_layer: 用于回环验证的注意力层编号（默认 21，即 Layer 22，从0开始计数）
    """
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, target_layer=21):
        super().__init__()

        # 核心骨干网络：基于 DINOv2-Large 的交替注意力聚合器（24层）
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, target_layer=target_layer)

        # 四个预测头，dim_in = 2*embed_dim 是因为拼接了帧内和全局注意力特征
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None                                       # 位姿编码 → 9维
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None  # 3D点 + 置信度
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None      # 深度 + 置信度
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None                   # 2D跟踪

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, compute_similarity=False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        # 如果输入没有 batch 维度 (S,3,H,W)，自动添加 → (1,S,3,H,W)
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        images = images.to(torch.bfloat16)
        # 运行交替注意力聚合器
        # 返回: aggregated_tokens_list (各层拼接特征), patch_start_idx (patch token起始位置),
        #       target_tokens (Layer 22 tokens, 用于图像检索), image_match_ratio (α_match, 论文§IV-B)
        aggregated_tokens_list, patch_start_idx, target_tokens, image_match_ratio = self.aggregator(images, compute_similarity)

        predictions = {}

        # 使用全精度运行各预测头（关闭自动混合精度），确保几何精度
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # 位姿预测头：输出 9 维编码 [T(3), Q(4), FoV(2)] (B,S,9)
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration（最后一次迭代的结果）
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                # 深度预测头：输出深度图 (B,S,H,W,1) 和置信度 (B,S,H,W)
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     predictions["world_points"] = pts3d
            #     predictions["world_points_conf"] = pts3d_conf

        # if self.track_head is not None and query_points is not None:
        #     track_list, vis, conf = self.track_head(
        #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
        #     )
        #     predictions["track"] = track_list[-1]  # track of the last iteration
        #     predictions["vis"] = vis
        #     predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        # target_tokens: Layer 22 的归一化 token（用于SALAD图像检索的特征增强）
        predictions["target_tokens"] = target_tokens  # for image retrieval feature extraction
        # image_match_ratio: α_match 匹配分数（论文§IV-B，用于回环验证，> 0.85 则确认回环）
        predictions["image_match_ratio"] = image_match_ratio

        return predictions

