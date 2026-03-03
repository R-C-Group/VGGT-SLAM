"""
graph.py — 位姿图（PoseGraph）模块

本模块实现了基于 GTSAM 的位姿图优化。核心特点是在 SL(4) 流形上操作。

SL(4) 流形说明：
- SL(4) 是行列式为 1 的 4x4 矩阵构成的特殊线性群
- 与传统 SLAM 使用的 SE(3)（刚体变换群）不同，SL(4) 是更广泛的变换群
- SL(4) 可以表示射影变换，包含尺度、倾斜等更丰富的变换
- 这使得系统能在射影空间中进行优化，更好地处理前馈网络预测的不完美性

位姿图包含：
- 节点（SL4 类型）：表示每帧的全局单应矩阵
- 边/因子：
  - PriorFactorSL4：先验约束（锚定第一帧到原点）
  - BetweenFactorSL4：相对变换约束
    - 子图内约束（inner_submap_noise）：同一子图中相邻帧之间
    - 子图间约束（intra_submap_noise）：不同子图的重叠帧之间
"""

import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# GTSAM 因子图组件
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4  # SL(4) 流形上的因子
from gtsam.symbol_shorthand import X  # 节点符号简写（X(key) 生成唯一节点标识）

from vggt_slam.slam_utils import decompose_camera, normalize_to_sl4

class PoseGraph:
    """
    位姿图：基于 GTSAM 的 SL(4) 因子图。

    管理所有关键帧的全局位姿（以 SL(4) 单应矩阵表示），
    并通过相对变换约束（BetweenFactor）和先验约束（PriorFactor）
    构建非线性因子图，使用 Levenberg-Marquardt 算法进行优化。

    Attributes:
        graph: GTSAM 非线性因子图
        values: GTSAM 变量值集合（存储所有 SL4 节点的当前估计值）
        inner_submap_noise: 子图内相邻帧约束的噪声模型
        intra_submap_noise: 子图间重叠帧约束的噪声模型
        anchor_noise: 先验约束的噪声模型（非常紧，几乎固定）
        auto_cal_H_mats: 自动校准单应矩阵字典（用于额外的位姿校正）
    """
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()      # GTSAM 非线性因子图
        self.values = Values()                    # 变量值集合

        # 噪声模型定义：
        # SL(4) 有 15 个自由度（4x4 矩阵 - 1 个行列式约束）
        # 因此噪声向量维度为 15
        inner_noise = 0.05*np.ones(15, dtype=float)  # 子图内约束噪声（较小 = 较信任）
        intra_noise = 0.05*np.ones(15, dtype=float)   # 子图间约束噪声
        self.inner_submap_noise = noiseModel.Diagonal.Sigmas(inner_noise)
        self.intra_submap_noise = noiseModel.Diagonal.Sigmas(intra_noise)
        # 先验约束噪声（极小值 1e-6，几乎完全固定第一帧）
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        self.initialized_nodes = set()    # 已初始化的节点集合（防止重复添加）
        self.num_loop_closures = 0 # Just used for debugging and analysis

        self.auto_cal_H_mats = dict()  # Store homographies estimated by auto-calibration
        # 自动校准单应矩阵：存储由外部校准过程估计的额外变换

    def add_homography(self, key, global_h):
        """
        Add a new homography node to the graph.
        向位姿图添加新的单应矩阵节点。

        Args:
            key: 节点编号（全局唯一的整数）
            global_h: 4x4 全局单应矩阵
        """
        # print("det(global_h)", np.linalg.det(global_h))
        # global_h = normalize_to_sl4(global_h)
        key = X(key)  # 转换为 GTSAM 符号键
        if key in self.initialized_nodes:
            print(f"SL4 {key} already exists.")
            return
        self.values.insert(key, SL4(global_h))  # 插入 SL4 值
        self.initialized_nodes.add(key)

    def add_between_factor(self, key1, key2, relative_h, noise):
        """
        Add a relative SL4 constraint between two nodes.
        添加两个节点之间的相对 SL(4) 约束。

        数学含义：约束 key2 = key1 @ relative_h
        即 relative_h = inv(key1) @ key2

        Args:
            key1: 第一个节点编号
            key2: 第二个节点编号
            relative_h: 两节点之间的相对单应矩阵 (4x4)
            noise: 噪声模型（表示约束的不确定性）
        """
        # relative_h = normalize_to_sl4(relative_h)
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(gtsam.BetweenFactorSL4(key1, key2, SL4(relative_h), noise))
    
    def add_prior_factor(self, key, global_h):
        """
        添加先验因子，将指定节点锚定到给定的单应矩阵。

        使用极紧的噪声模型（anchor_noise），使得该节点在优化中几乎不移动。
        通常用于将第一帧固定在世界坐标系原点。

        Args:
            key: 节点编号
            global_h: 先验单应矩阵 (4x4)
        """
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorSL4(key, SL4(global_h), self.anchor_noise))

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        获取指定节点的优化后 SL(4) 单应矩阵。

        如果存在自动校准矩阵，会将其与优化结果相乘。
        最终返回 auto_cal_H @ optimized_H 的 4x4 矩阵。

        :param node_id: The ID of the node.  节点编号
        :return: gtsam.SL4 homography of the node.  4x4 numpy 矩阵
        """

        auto_cal_H = np.eye(4)  # 默认无校准（单位矩阵）
        if node_id in self.auto_cal_H_mats:
            auto_cal_H  = self.auto_cal_H_mats[node_id]
        node_id = X(node_id)
        return auto_cal_H @ self.values.atSL4(node_id).matrix()

    def get_projection_matrix(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        获取指定节点的投影矩阵（单应矩阵的逆）。

        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """
        homography = self.get_homography(node_id)
        projection_matrix = np.linalg.inv(homography)
        return projection_matri

    
    def optimize(self, verbose=False):
        """
        Optimize the graph with Levenberg–Marquardt and print per-factor errors.
        使用 Levenberg-Marquardt 算法优化位姿图。

        LM 优化器会迭代地最小化所有因子的总误差，
        在 SL(4) 流形上调整每个节点的单应矩阵。

        Args:
            verbose: 是否打印详细的逐因子误差信息
        """
        # Optional verbosity settings
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)

        # --- Initial total error ---
        # 输出优化前的总误差
        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        # --- Per-factor initial error ---
        # 可选：输出每个因子的初始误差
        if verbose:
            print("\nInitial per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

            keys = [gtsam.DefaultKeyFormatter(k) for k in factor.keys()]
            print(f"Factor {i} connects to {keys} with error {e:.6f}")

        # --- Optimize ---
        # 执行 LM 优化
        result = optimizer.optimize()

        # --- Final total error ---
        final_error = self.graph.error(result)
        # print(f"\nFinal total error: {final_error:.6f}")

        # --- Per-factor final error ---
        # 可选：输出每个因子的最终误差
        if verbose:
            print("\nFinal per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        # --- Store optimized values ---
        # 存储优化后的结果（替换原始估计值）
        self.values = result


    def print_estimates(self):
        """Print the optimized poses. 打印所有优化后的位姿。"""
        for key in sorted(self.initialized_nodes):
            print(f"Homography{key}:\n{self.values.atSL4(key)}\n")
    
    def increment_loop_closure(self):
        """Increment the loop closure count. 递增回环闭合计数器。"""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures. 获取回环闭合次数。"""
        return self.num_loop_closures

    def update_all_homographies(self, map, auto_cal_H_mats):
        """
        更新所有节点的自动校准单应矩阵。

        当外部校准过程（如自动标定）产生新的校正矩阵时调用。
        将校正矩阵存储到 auto_cal_H_mats 字典中，
        后续 get_homography 会自动应用这些校正。

        Args:
            map: GraphMap 实例
            auto_cal_H_mats: 校准单应矩阵列表（与位姿数量一一对应）
        """
        count = 0
        for submap in map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                # 存储校准矩阵的逆（因为 get_homography 中是左乘）
                self.auto_cal_H_mats[id] = np.linalg.inv(auto_cal_H_mats[count])
                count += 1
        assert count == len(auto_cal_H_mats), "Number of auto-calibration homographies does not match number of poses in the map."