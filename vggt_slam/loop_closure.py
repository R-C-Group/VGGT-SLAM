"""
loop_closure.py — 回环检测模块

本模块实现了基于图像检索的回环检测功能。

核心组件：
1. ImageRetrieval 类：基于 SALAD (DINOv2 + Salad) 的全局图像检索模型
   - SALAD 使用 DINOv2 视觉特征提取全局描述子
   - 通过描述子之间的 L2 距离判断图像相似度
2. LoopMatch：回环匹配的数据结构（命名元组）
3. LoopMatchQueue：优先队列，维护最佳的 N 个回环匹配

回环检测流程：
1. 对当前子图的每个帧，提取 SALAD 全局描述子
2. 在历史子图的描述子数据库中搜索最相似的帧
3. 如果相似度超过阈值，将其作为潜在回环
4. 使用优先队列保留最佳的 max_loop_closures 个回环
5. 后续在 Solver 中，会对回环帧对运行 VGGT 推理并验证匹配率
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import heapq
from typing import NamedTuple
import torchvision.transforms as T
import os

from salad.eval import load_model # load salad 加载 SALAD 模型

device = 'cuda'

# 图像预处理工具
tensor_transform = T.ToPILImage()  # 张量 → PIL 图像
# 反归一化变换（将 [-1,1] 范围恢复到 [0,1]）
denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

def input_transform(image_size=None):
    """
    创建 SALAD 模型的图像预处理管道。

    包括：
    1. 可选的 Resize（调整到指定尺寸）
    2. ToTensor（转为张量）
    3. Normalize（ImageNet 均值和标准差归一化）

    Args:
        image_size: 目标图像尺寸 (H, W)，None 表示不调整大小

    Returns:
        Compose 预处理管道
    """
    MEAN = [0.485, 0.456, 0.406]  # ImageNet 均值
    STD = [0.229, 0.224, 0.225]    # ImageNet 标准差
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)

class LoopMatch(NamedTuple):
    """
    回环匹配数据结构。

    Attributes:
        similarity_score: 相似度得分（L2 距离，越小越相似）
        query_submap_id: 查询子图的 ID
        query_submap_frame: 查询子图中的帧索引
        detected_submap_id: 检测到的匹配子图 ID
        detected_submap_frame: 检测到的匹配子图中的帧索引
    """
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int

class LoopMatchQueue:
    """
    回环匹配优先队列。

    使用最大堆维护最佳的 N 个回环匹配（L2 距离最小的）。
    通过取反相似度分数将 Python 的最小堆转换为最大堆。

    Attributes:
        max_size: 队列最大容量
        heap: 堆数据结构
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Simulated max-heap by negating scores 通过取反实现最大堆

    def add(self, match: LoopMatch):
        """
        添加一个回环匹配到队列中。

        如果队列未满，直接添加。
        如果队列已满，替换掉最差的匹配（L2 距离最大的）。

        Args:
            match: LoopMatch 实例
        """
        # Negate similarity_score to turn min-heap into max-heap
        # 取反相似度分数，将 Python 的最小堆模拟为最大堆
        item = (-match.similarity_score, match)
        # item = (-match.detected_submap_id, match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            # Push new element and remove the largest (i.e., smallest negated)
            # 推入新元素并弹出最大值（即最差匹配）
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        """Return sorted list of matches (lowest value first)
        返回按相似度排序的匹配列表（最佳匹配在前）。"""
        return [match for _, match in sorted(self.heap, reverse=True)]
        

class ImageRetrieval:
    """
    基于 SALAD 的图像检索模块。

    SALAD（Sinkhorn Algorithm for Locally-Aggregated Descriptors）是一种
    基于 DINOv2 视觉特征的全局图像检索方法。

    工作流程：
    1. 使用 DINOv2 提取图像的局部特征
    2. 通过 SALAD 聚合为全局描述子
    3. 使用 L2 距离进行图像匹配

    Attributes:
        model: SALAD 检索模型
        transform: 图像预处理管道
    """
    def __init__(self, input_size=224):
        """
        初始化图像检索模块。

        Args:
            input_size: 输入图像大小（SALAD 要求 224x224）
        """
        # 从缓存目录加载预训练的 SALAD 模型
        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()  # 设置为评估模式
        self.transform = input_transform((input_size, input_size))

    def get_single_embeding(self, cv_img):
        """
        计算单张图像的 SALAD 全局描述子。

        Args:
            cv_img: 图像张量 (3, H, W)

        Returns:
            全局描述子向量
        """
        with torch.no_grad():
            pil_img = self.transform(tensor_transform(cv_img))
            return self.model(pil_img.to(device))

    def get_batch_descriptors(self, imgs):
        """
        批量计算图像的 SALAD 全局描述子。

        Args:
            imgs: 图像批量张量 (B, C, H, W)

        Returns:
            全局描述子批量 (B, D)
        """
        # Expecting imgs to be a batch of images (B, C, H, W)
        with torch.no_grad():
            pil_imgs = [tensor_transform(img) for img in imgs]  # Convert each tensor to PIL Image
            imgs = torch.stack([self.transform(img) for img in pil_imgs])  # Apply transform and stack
            return self.model(imgs.to(device))
    
    def get_all_submap_embeddings(self, submap):
        """
        计算子图中所有帧的 SALAD 全局描述子。

        Args:
            submap: Submap 实例

        Returns:
            所有帧的描述子 (S, D)
        """
        # Frames is np array of shape (S, 3, H, W)
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(self, map, submap, max_similarity_thres = 0.80, max_loop_closures = 0):
        """
        在历史子图中搜索回环闭合。

        对当前子图的每个帧，在全局地图的历史帧中搜索最相似的帧。
        如果 L2 距离低于阈值，则认为检测到潜在回环。

        Args:
            map: GraphMap 实例（包含所有历史子图）
            submap: 当前子图
            max_similarity_thres: L2 距离阈值（低于此值视为回环）
            max_loop_closures: 最大回环数量（0 表示不检测回环）

        Returns:
            LoopMatch 列表（按相似度排序，最佳匹配在前）
        """
        matches_queue = LoopMatchQueue(max_size=max_loop_closures)
        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            # 在历史子图中搜索最相似的帧
            best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(query_vector, submap.get_id(), ignore_last_submap=True)
            if best_score < max_similarity_thres:
                # L2 距离低于阈值，记录为潜在回环
                new_match_data = LoopMatch(best_score, submap.get_id(), query_id, best_submap_id, best_frame_id)
                matches_queue.add(new_match_data)
            query_id += 1
        
        return matches_queue.get_matches()