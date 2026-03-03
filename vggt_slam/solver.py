"""
solver.py — VGGT-SLAM 核心求解器模块

本模块实现了 Solver 类，是整个 VGGT-SLAM 系统的核心管理器。
主要职责包括：
1. 调用 VGGT 模型进行前馈推理（深度、位姿、3D点、置信度）
2. 管理子图（Submap）的创建和属性设置
3. 构建位姿图（PoseGraph），添加节点和约束边
4. 估计子图间的尺度因子（SL(4) 流形上的尺度对齐）
5. 检测回环闭合（基于 SALAD 图像检索）并添加回环约束
6. 管理3D点云的可视化（通过 Viser 查看器）
7. 可选的 CLIP 语义嵌入计算
"""

import numpy as np
import cv2
import gtsam                      # GTSAM：因子图优化库
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d              # Open3D：3D点云处理和可视化
from termcolor import colored      # 彩色终端输出
from scipy.linalg import rq       # RQ分解（用于投影矩阵分解）

# VGGT 第三方库中的几何工具函数
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri   # 将位姿编码转换为外参和内参矩阵
from vggt.utils.load_fn import load_and_preprocess_images       # 图像加载和预处理

# VGGT-SLAM 内部模块
from vggt_slam.slam_utils import compute_image_embeddings, Accumulator
from vggt_slam.loop_closure import ImageRetrieval   # 图像检索（用于回环检测）
from vggt_slam.frame_overlap import FrameTracker    # 光流帧跟踪器（关键帧选择）
from vggt_slam.map import GraphMap                  # 全局地图管理器
from vggt_slam.submap import Submap                 # 子图数据结构
from vggt_slam.graph import PoseGraph               # 位姿图（因子图）
from vggt_slam.scale_solver import estimate_scale_pairwise  # 成对尺度估计
from vggt_slam.viewer import Viewer                 # Viser 3D 可视化查看器

DEBUG = False  # 调试标志：启用后显示额外的可视化和日志

def debug_visualize(pcd1_points, pcd2_points):
    """
    调试用：将两组3D点云分别以红色和蓝色可视化。
    用于检查子图间点云对齐的效果。

    Args:
        pcd1_points: 第一组3D点 (N, 3)，红色显示
        pcd2_points: 第二组3D点 (M, 3)，蓝色显示
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    """
    VGGT-SLAM 核心求解器。

    协调以下功能：
    - VGGT 模型推理
    - 子图创建与管理
    - 位姿图构建（SL(4) 流形上的因子图）
    - 子图间尺度对齐
    - 回环检测与闭合
    - 3D可视化
    """
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,      # 回环检测图像检索的相似度阈值
        vis_voxel_size: float = None  # 可视化时的体素降采样大小，None表示不降采样
    ):
        
        self.init_conf_threshold = init_conf_threshold  # 置信度过滤百分位数阈值
        self.vis_voxel_size = vis_voxel_size

        self.viewer = Viewer()                # Viser 3D 可视化服务器

        self.flow_tracker = FrameTracker()    # 光流帧跟踪器（用于关键帧选择）
        self.map = GraphMap()                 # 全局地图（存储所有子图）
        self.graph = PoseGraph()              # 位姿图（GTSAM 因子图）

        self.image_retrieval = ImageRetrieval()  # 图像检索模块（SALAD 模型）
        self.current_working_submap = None       # 当前正在构建的子图

        self.lc_thres = lc_thres            # 回环检测阈值

        self.temp_count = 0                  # 临时计数器
        self.vggt_timer = Accumulator()      # VGGT 模型推理计时器
        self.loop_closure_timer = Accumulator()  # 回环检测计时器
        self.clip_timer = Accumulator()      # CLIP 语义嵌入计时器

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        """
        在 Viser 3D 场景中设置/更新一个点云。
        如果设置了 vis_voxel_size，会先进行体素降采样以减少渲染负载。

        Args:
            points_in_world_frame: 世界坐标系下的3D点 (N, 3)
            points_colors: 对应的颜色 (N, 3)，值域 [0, 255]
            name: 点云在场景中的标识名
            point_size: 点的渲染大小
        """
        if self.vis_voxel_size is not None:
            # 使用 Open3D 进行体素降采样
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        # 将点云添加到 Viser 场景中
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        """将指定子图的点云转换到世界坐标系并添加到可视化场景中。"""
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        """将指定子图的所有相机位姿添加到可视化场景中。"""
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames()
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        """更新所有子图的可视化（点云和位姿），通常在回环修正后调用。"""
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        """仅更新最新子图的可视化，用于无回环时的增量更新。"""
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        """
        将子图变换为标准坐标系（第一帧相机矩阵为单位阵 [I | 0]）。

        这是一种归一化操作，使得子图在其自身局部坐标系下表示，
        第一帧的投影矩阵为单位矩阵。

        Args:
            proj_mat_world_to_cam: 世界到相机的投影矩阵 (S, 4, 4)
            world_points: 世界坐标系下的3D点 (S, H, W, 3)

        Returns:
            变换后的投影矩阵和点云
        """
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        # 对所有相机矩阵施加变换，使第一帧变为 [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        # 对点云施加相同的变换
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            # 将3D点转为齐次坐标 (N, 4)
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T # TODO Dominic check if we want to use P_prior here
            # 齐次坐标归一化
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
        """
        在位姿图中添加一条边（约束）。

        核心功能：
        1. 估计相邻子图之间的尺度因子（通过重叠帧的点云对比）
        2. 计算新子图在世界坐标系下的首帧单应矩阵
        3. 添加子图间（intra-submap）约束因子
        4. 添加子图内（inner-submap）连续帧约束因子
        5. 对于第一个子图，添加先验因子（锚定原点）

        数学原理：
        - 在 SL(4) 流形上操作（4x4 单应矩阵，行列式归一化）
        - 子图间通过重叠帧的点云匹配估计尺度因子
        - 尺度因子 s 应用为 diag(s, s, s, 1) 的缩放矩阵

        Args:
            submap_id_curr: 当前子图的 ID（全局偏移量）
            frame_id_curr: 当前子图中的帧索引
            submap_id_prev: 前一子图的 ID（None 表示首个子图）
            frame_id_prev: 前一子图中用于对齐的帧索引
            is_loop_closure: 是否为回环闭合约束
        """
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)  # 世界到子图的单应矩阵（初始为单位阵）

        if submap_id_prev is not None:
            # ---- 存在前一子图：需要估计尺度并建立约束 ----
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            # 获取前一子图，用于尺度估计
            prior_submap = self.map.get_submap(submap_id_prev)

            # 获取重叠帧的置信度掩码，用于筛选高质量的对应点
            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            # 只保留两帧中置信度都超过阈值的点
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
            good_mask = good_mask.reshape(-1)

            # 如果高质量重叠点太少，逐步放宽约束
            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points. 
                    good_mask = (prior_conf > 0).reshape(-1)

            # 计算从前一子图坐标系到当前子图坐标系的变换
            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            # 将当前子图的点云变换到前一子图的坐标系下
            t1 = (P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            # 通过点对距离的中值估计尺度因子
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output)
            scale_factor = scale_factor_est_output[0]
            # 构建尺度缩放矩阵 diag(s, s, s, 1)
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            # 计算新子图在世界坐标系下的首帧投影矩阵
            # H_overlap = inv(P_prior_last) @ P_current_first @ H_scale
            H_overlap_prior_overlap_current = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0] @ H_scale
            # 通过前一子图的世界位姿链式传递得到当前子图的世界位姿
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            # 将新子图的第一个节点添加到位姿图中
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint.
            # 添加子图间约束因子（连接前一子图的最后一帧和当前子图的第一帧）
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            # ---- 第一个子图：设置全局原点 ----
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            # 添加原点节点（单位矩阵）
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            # 添加先验因子，将第一帧锚定在原点
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        # 回环闭合约束只需要子图间约束，不需要添加内部连续帧约束
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        # 添加子图内部连续帧之间的约束
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue  # 第一帧已经在上面处理过

            # 计算相邻帧之间的相对变换
            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose) # TODO Dominic, no need to take the inverse twice, just use cam_to_world
            # 通过链式传递计算当前帧的世界位姿
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            # 将当前帧添加为位姿图节点
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            # 添加子图内连续帧之间的约束因子
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

    def add_points(self, pred_dict):
        """
        将 VGGT 模型的预测结果添加到全局地图和位姿图中。

        主要步骤：
        1. 从预测字典中提取图像、外参、内参、深度图和置信度
        2. 利用深度图和相机参数反投影得到3D点云
        3. 设置子图的属性（位姿、点云、颜色、置信度等）
        4. 将子图添加到全局地图
        5. 添加位姿图约束（子图间 + 子图内）
        6. 如果有回环闭合，创建回环子图并添加回环约束

        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        # 解包预测字典
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W) 输入图像
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4) 外参矩阵（世界到相机）
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3) 内参矩阵

        detected_loops = pred_dict["detected_loops"]  # 检测到的回环列表

        depth_map = pred_dict["depth"]  # (S, H, W, 1) 深度图
        conf = pred_dict["depth_conf"]  # (S, H, W) 深度置信度

        # 利用深度图、外参和内参反投影得到世界坐标系下的3D点云
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # 将图像从 (S,3,H,W) 转换为 (S,H,W,3) 的 uint8 格式（用于可视化）
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        # 计算相机坐标系到世界坐标系的变换（SE3 解析逆）
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        # 构建 4x4 投影矩阵（将 3x3 内参扩展为 4x4）
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)  # 世界到相机的变换矩阵


        # 获取前一个子图的 ID（忽略回环子图）
        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0           # 当前子图中的帧索引（从 0 开始）
        frame_id_prev = None

        first_edge = submap_id_prev is None  # 是否为第一条边（第一个子图）

        if not first_edge:
            # 获取前一子图的最后一个非回环帧索引（用于重叠对齐）
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        # 设置子图属性并将子图添加到全局地图
        self.current_working_submap.add_all_poses(world_to_cam)           # 添加所有帧的位姿
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, K_4x4)  # 添加3D点、颜色、置信度
        self.current_working_submap.set_conf_masks(conf)                   # 设置置信度掩码
        self.map.add_submap(self.current_working_submap)                   # 添加到全局地图

        # Add all constraints for the new submap.
        # 为新子图添加所有位姿图约束（子图间 + 子图内）
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        # Add in loop closures if any were detected.
        # 处理检测到的回环闭合
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            # 计算回环帧对的相机变换和投影矩阵
            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"]) 
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1) 回环帧深度图
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W) 回环帧深度置信度

            intrinsics_cam = pred_dict["intrinsic_lc"]
            
            # 反投影得到回环帧的3D点云
            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            # 创建回环子图（一种特殊的子图，仅包含查询帧和检索帧）
            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)  # 标记为回环子图
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc)
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)
            self.map.add_submap(lc_submap)

            # 添加回环约束：
            # 1. 查询帧 → 回环子图第一帧（正向约束）
            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            # 2. 检索到的历史帧 ← 回环子图第二帧（反向回环约束）
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)

    def sample_pixel_coordinates(self, H, W, n):
        """
        在图像平面上随机采样 n 个像素坐标。

        Args:
            H: 图像高度
            W: 图像宽度
            n: 采样点数量

        Returns:
            pixel_coords: (n, 2) 的张量，每行为 (y, x)
        """
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess):
        """
        运行 VGGT 模型推理并检测回环闭合。

        这是每个子图处理的核心函数，主要步骤：
        1. 加载和预处理图像
        2. 创建新子图并计算图像检索向量（SALAD）
        3. 可选地计算 CLIP 语义嵌入
        4. 运行 VGGT 前馈模型推理
        5. 检索图像数据库进行回环检测
        6. 如果检测到回环，对回环帧对运行 VGGT 推理
        7. 将位姿编码转换为外参和内参矩阵

        Args:
            image_names: 图像文件路径列表
            model: VGGT 模型实例
            max_loops: 最大回环数量
            clip_model: CLIP 模型（可为 None）
            clip_preprocess: CLIP 图像预处理（可为 None）

        Returns:
            predictions: 包含所有预测结果的字典
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        # 加载和预处理图像（调整大小、归一化等）
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        # 根据 GPU 计算能力选择精度（Ampere 及以上用 bfloat16，否则用 float16）
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # First submap so set new pcd num to 0
        # 确定新子图的全局 ID（基于位姿图中已有节点的最大编号）
        if self.map.get_largest_key() is None:
            new_pcd_num = 0  # 第一个子图
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        # 创建新子图并设置其属性
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)                    # 添加图像帧张量
        new_submap.set_frame_ids(image_names)                # 从文件名提取帧 ID
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)  # 最后一个非回环帧索引
        # 计算子图所有帧的图像检索向量（用于回环检测）
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)                # 保存图像文件名

        # 可选：计算 CLIP 语义嵌入向量（用于开放集语义搜索）
        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        # ---- 运行 VGGT 前馈模型推理 ----
        with torch.no_grad():
            t1 = time.time()
            with self.vggt_timer:
                predictions = model(images)  # 返回深度、位姿编码、置信度等
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        # Check for loop closures and add retrieval vectors from new submap to the database
        # ---- 回环检测 ----
        predictions_lc = None
        with self.loop_closure_timer:
            # 在历史子图中搜索与当前子图最相似的帧
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            # 获取检索到的历史帧
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                # 将当前子图的查询帧和检索到的帧组成一对，运行 VGGT 推理
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)
                predictions_lc = model(lc_frames, compute_similarity=True)
                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame), 
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        # ---- 将位姿编码转换为标准的外参和内参矩阵 ----
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops
        
        # ---- 验证回环质量并处理 ----
        if predictions_lc is not None:
            # 检查图像匹配比率（VGGT 输出的相似度指标）
            image_match_ratio = predictions_lc["image_match_ratio"]
            if image_match_ratio < 0.85:
                # 匹配率太低，放弃此回环
                print(colored("Loop closure image match ratio too low, skipping loop closure", "red"))
                predictions_lc = None # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
            else:
                # 回环质量良好，处理回环预测结果
                self.graph.increment_loop_closure()
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
                predictions["extrinsic_lc"] = extrinsic_lc
                predictions["intrinsic_lc"] = intrinsic_lc
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]

        # 将所有张量预测结果转换为 numpy 数组（移除 batch 维度）
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    
        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]  # 回环帧对
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        return predictions