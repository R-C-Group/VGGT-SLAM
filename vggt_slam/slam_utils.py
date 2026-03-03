"""
slam_utils.py — SLAM 工具函数集合

本模块包含 VGGT-SLAM 系统中使用的各种通用工具函数：
1. 图像序列处理（排序、降采样）
2. 投影矩阵分解（RQ分解提取 K, R, t）
3. CLIP 语义嵌入计算（图像和文本）
4. 余弦相似度计算
5. SL(4) 归一化
6. 有向包围盒（OBB）计算（基于PCA）
7. 掩码叠加可视化
8. 累积计时器
"""

import os
import re
import numpy as np
import matplotlib
import scipy
import time
from PIL import Image
from torchvision import transforms as TF
import torch


def slice_with_overlap(lst, n, k):
    """
    将列表按步长切片，支持重叠。

    Args:
        lst: 输入列表
        n: 每个切片的大小
        k: 重叠数量（相邻切片共享 k 个元素）

    Returns:
        切片列表的列表
    """
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    """
    按文件名中的数字对图像路径列表排序。

    例如：["frame_3.jpg", "frame_1.jpg", "frame_2.jpg"]
    排序后：["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]

    支持整数和小数形式的数字（如 "1654685.123456.png"）。

    Args:
        image_paths: 图像文件路径列表

    Returns:
        排序后的路径列表
    """
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits followed immediately by a dot and the extension
        # 查找紧接在扩展名之前的数字
        match = re.search(r'\d+(?:\.\d+)?(?=\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')

    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    通过保留每第 downsample_factor 个图像来降采样图像列表。
    
    Args:
        image_names (list of str): List of image filenames.
            图像文件名列表
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.
            降采样因子。例如 2 表示保留每隔一个图像。

    Returns:
        list of str: Downsampled list of image filenames.
            降采样后的图像文件名列表
    """
    return image_names[::downsample_factor]

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    将 3x4 或 4x4 相机投影矩阵 P 分解为内参 K、旋转 R 和平移 t。

    分解过程：
    1. 如果输入为 4x4，先归一化并截取前 3 行
    2. 对 P 的左 3x3 部分进行 RQ 分解：M = K @ R
    3. 确保 K 对角线元素为正（通过翻转列/行符号）
    4. 计算平移向量 t = -R @ K^{-1} @ P[:,3]

    Args:
        P: 3x4 或 4x4 投影矩阵
        no_inverse: 如果为 True，不对 R 取逆

    Returns:
        K: 3x3 归一化内参矩阵
        R: 3x3 旋转矩阵
        t: 3D 平移向量
        scale: K[2,2] 的原始值（用于归一化前的尺度参考）
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]  # 归一化 4x4 矩阵
        P = P[0:3, :]      # 截取前 3 行

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    # 取左 3x3 部分
    M = P[:, :3]

    # RQ decomposition
    # RQ 分解：M = K @ R，其中 K 为上三角（内参），R 为正交（旋转）
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    # 确保内参矩阵对角线元素为正
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    # print("Scale factor from K[2,2]:", scale)
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]  # 计算平移：t = -R @ K^{-1} @ p4
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale  # 归一化内参矩阵使 K[2,2] = 1
    
    return K, R, t, scale

def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    """
    使用 CLIP 模型计算图像的语义嵌入向量。

    Args:
        model: CLIP 模型实例
        preprocess: CLIP 图像预处理变换
        image_paths: 图像文件路径列表
        batch_size: 批处理大小
        device: 计算设备

    Returns:
        归一化后的图像嵌入向量 (N, D)，numpy 数组
    """
    all_embs = []

    # Load all images into memory (PIL -> tensor)
    # 加载所有图像并进行预处理
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    # Stack into a single tensor
    imgs = torch.stack(imgs).to(device)

    # Loop over batches
    # 分批计算嵌入向量
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 归一化
            all_embs.append(emb.cpu())

    # Combine into one (N, D) array
    return torch.cat(all_embs, dim=0).numpy()

def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    """
    使用 CLIP 模型计算文本的语义嵌入向量。

    Args:
        clip_model: CLIP 模型实例
        tokenizer: CLIP 文本分词器
        text: 文本查询字符串
        device: 计算设备

    Returns:
        归一化后的文本嵌入向量 (1, D)，numpy 数组
    """
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)     # 分词
        text_emb = clip_model.encode_text(text_tokens)  # 编码
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 归一化
        return text_emb.cpu().numpy()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    计算两个向量之间的余弦相似度。

    cos_sim = (a · b) / (||a|| * ||b||)

    Args:
        a, b: 向量或向量数组

    Returns:
        余弦相似度值
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    将 4x4 单应矩阵归一化到 SL(4) 群（行列式为 1）。

    归一化方法：H_normalized = H / det(H)^{1/4}
    这确保了 det(H_normalized) = 1。

    Args:
        H: 4x4 单应矩阵

    Returns:
        归一化后的 SL(4) 矩阵
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) for a Nx3 point cloud.
    为 Nx3 的3D点云计算有向包围盒（OBB）。

    算法步骤：
    1. 计算点云质心
    2. 对中心化的点云进行 PCA（主成分分析）
    3. 将点云投影到 PCA 坐标系（主轴方向）
    4. 在 PCA 坐标系中计算轴对齐包围盒
    5. 将包围盒中心转换回世界坐标系

    Returns:
        center      : (3,) world-space center of OBB  世界坐标系下的 OBB 中心
        extent      : (3,) lengths of OBB along its principal axes  OBB 沿主轴的边长
        rotation    : (3,3) rotation matrix (columns = principal axes)  旋转矩阵（列=主轴方向）
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3 points"

    # Remove NaN/inf if any
    # 移除无效点（NaN 或 inf）
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    # 1. Compute centroid
    # 1. 计算质心
    centroid = points.mean(axis=0)

    # 2. PCA on centered points
    # 2. 对中心化的点进行 PCA
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)  # 计算协方差矩阵

    # Eigen decomposition (sorted by eigenvalue descending)
    # 特征值分解（按特征值降序排列）
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotation = eigvecs  # columns = principal axes (R) 列=主轴方向

    # 3. Project points to PCA frame
    # 3. 将点投影到 PCA 坐标系
    points_local = centered @ rotation

    # 4. Compute min/max in PCA frame → box extents
    # 4. 在 PCA 坐标系中计算包围盒边界
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner  # 各轴方向的边长

    # 5. Compute box center (local), then world
    # 5. 计算包围盒中心（局部 → 世界）
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation

def overlay_masks(image, masks):
    """
    在图像上叠加彩色分割掩码。

    使用 rainbow 色表为每个掩码分配不同颜色，
    并以半透明方式叠加在原图上。

    Args:
        image: PIL Image 对象
        masks: 分割掩码张量 (N, 1, H, W) 或 (N, H, W)

    Returns:
        叠加掩码后的 RGBA PIL Image
    """
    image = image.convert("RGBA")

    # masks: (N, 1, H, W) or (N, H, W)
    masks = (255 * masks.cpu().numpy()).astype(np.uint8)

    # 为每个掩码分配不同的颜色
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        # Now mask is shape (H, W)

        mask = Image.fromarray(mask)
        # 创建半透明彩色叠加层
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))  # 50% 透明度
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image

class Accumulator:
    """
    累积计时器，用于统计各个阶段的总耗时。

    支持 with 语句的上下文管理器模式：
        timer = Accumulator()
        with timer:
            # 要计时的代码
        print(timer.total_time)  # 累积的总时间（秒）
    """
    def __init__(self):
        self.total_time = 0  # 累积总时间（秒）

    def __enter__(self):
        """进入计时上下文，记录开始时间。"""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        """退出计时上下文，累加耗时。"""
        self.total_time += (time.perf_counter() - self.start)
