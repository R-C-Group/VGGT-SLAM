"""
scale_solver.py — 成对尺度估计模块

本模块实现了子图之间的尺度因子估计。

问题背景：
- VGGT 模型在每个子图内可以准确估计相对深度和3D几何
- 但不同子图之间存在尺度不一致问题（单目 SLAM 的固有问题）
- 需要通过重叠帧的点云对比来估计子图间的尺度因子

算法原理：
- 对于两个子图的重叠帧，VGGT 分别预测了它们的3D点云
- 对应的3D点到原点的距离之比就是尺度因子
- 使用中值（median）而非均值来提高鲁棒性（减少离群点影响）

数学表达：
  scale = median(||Y_i|| / ||X_i||)
  其中 X_i 和 Y_i 是两个子图中对应的3D点
"""

import numpy as np
import open3d as o3d

def debug_visualize(pcd1_points, pcd2_points):
    """
    调试用：将两组3D点云分别以红色和蓝色可视化。
    用于检查尺度对齐后的效果。

    Args:
        pcd1_points: 第一组点 (N, 3)，红色显示
        pcd2_points: 第二组点 (N, 3)，蓝色显示
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red 红色

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue 蓝色

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

def estimate_scale_pairwise(X, Y, DEBUG=False):
    """
    估计两组3D点之间的尺度因子。

    给定两组对应的3D点 X 和 Y（来自两个子图的重叠帧），
    通过比较每对对应点到原点的距离来估计尺度因子。

    使用中值估计以提高对离群点的鲁棒性。

    数学公式：
        scales[i] = ||Y_i|| / ||X_i||
        scale = median(scales)

    使得 scale * X ≈ Y

    Args:
        X: 第一组3D点 (N, 3)，来自当前子图
        Y: 第二组3D点 (N, 3)，来自前一子图
        DEBUG: 是否进行可视化调试

    Returns:
        (scale, None): 估计的尺度因子和占位符（保留接口一致性）
    """
    assert X.shape == Y.shape  # 两组点必须一一对应
    # 计算每个点到原点的距离
    x_dists = np.linalg.norm(X, axis=1)
    y_dists = np.linalg.norm(Y, axis=1)
    # 计算每对点的尺度比
    scales = y_dists / x_dists
    # 取中值作为鲁棒的尺度估计
    scale = np.median(scales)

    if DEBUG:
        # 可视化对齐后的效果
        debug_visualize(X*scale, Y)

    return scale, None