"""
submap.py — 子图（Submap）数据结构模块

本模块定义了 Submap 类，用于管理 SLAM 系统中单个子图的所有数据。
每个子图包含一组连续的关键帧，以及它们对应的：
- 相机位姿（世界到相机的变换矩阵）
- 3D点云（各帧的像素级3D点）
- 颜色信息
- 深度置信度
- 投影矩阵（4x4 内参矩阵）
- 图像检索向量（用于回环检测）
- 语义嵌入向量（用于开放集搜索）

子图是 VGGT-SLAM 的核心数据单元：
- 每个子图包含 submap_size 个关键帧
- 相邻子图之间有重叠帧用于尺度对齐
- 回环子图（lc_submap）仅包含2帧（查询帧 + 检索帧）
"""

import re
import os
import cv2
import torch
import numpy as np
import scipy
import open3d as o3d
from vggt_slam.slam_utils import decompose_camera  # 投影矩阵分解工具

class Submap:
    """
    子图数据结构，存储一组连续关键帧的所有相关信息。

    Attributes:
        submap_id: 子图的全局唯一 ID（也是位姿图中第一帧的节点编号）
        poses: 所有帧的世界到相机变换矩阵 (N, 4, 4)
        frames: 图像帧张量 (N, 3, H, W)
        proj_mats: 4x4 投影矩阵（包含内参） (N, 4, 4)
        pointclouds: 各帧的3D点云 (N, H, W, 3)
        colors: 各帧的颜色 (N, H, W, 3)
        conf: 深度置信度 (N, H, W)
        conf_masks: 用于过滤的置信度掩码 (N, H, W)
        conf_threshold: 置信度过滤阈值（自动计算的百分位数值）
        retrieval_vectors: 图像检索向量（SALAD，用于回环检测）
        semantic_vectors: 语义嵌入向量（CLIP，用于开放集搜索）
        is_lc_submap: 是否为回环子图
        last_non_loop_frame_index: 最后一个非回环帧的索引
    """
    def __init__(self, submap_id):
        """
        初始化子图。

        Args:
            submap_id: 子图的全局唯一 ID。该 ID 同时作为位姿图中
                       第一帧的节点编号，因此必须在全局范围内唯一。
        """
        self.submap_id = submap_id
        self.R_world_map = None          # 世界到地图的旋转（暂未使用）
        self.poses = None                 # 所有帧的位姿 (N, 4, 4)
        self.frames = None                # 图像帧张量 (N, 3, H, W)
        self.proj_mats = None             # 4x4 投影矩阵 (N, 4, 4)
        self.retrieval_vectors = None     # 图像检索向量
        self.colors = None # (S, H, W, 3) 颜色数据
        self.conf = None # (S, H, W) 深度置信度
        self.conf_masks = None # (S, H, W) 置信度掩码
        self.conf_threshold = None        # 基于百分位数计算的置信度阈值
        self.pointclouds = None # (S, H, W, 3) 3D点云
        self.voxelized_points = None      # 体素化后的点云（缓存）
        self.last_non_loop_frame_index = None  # 最后一个非回环帧的索引
        self.frame_ids = None             # 从文件名提取的帧编号
        self.is_lc_submap = False         # 是否为回环闭合子图
        self.img_names = []               # 图像文件路径列表
        self.semantic_vectors = []        # CLIP 语义嵌入向量列表
    
    def set_lc_status(self, is_lc_submap):
        """设置该子图是否为回环闭合子图。"""
        self.is_lc_submap = is_lc_submap
    
    def add_all_poses(self, poses):
        """设置子图中所有帧的位姿（世界到相机变换矩阵）。"""
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics_inv):
        """
        设置子图的3D点云数据。

        Args:
            points: 3D点云 (S, H, W, 3)
            colors: 颜色 (S, H, W, 3)，值域 [0, 255]
            conf: 深度置信度 (S, H, W)
            conf_threshold_percentile: 置信度过滤百分位数
                例如 25 表示将置信度低于第 25 百分位数的点过滤掉
            intrinsics_inv: 4x4 投影矩阵（包含内参）(S, 4, 4)
        """
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        # 根据百分位数计算置信度阈值，加上微小值避免边界问题
        self.conf_threshold = np.percentile(self.conf, conf_threshold_percentile) + 1e-6
        self.proj_mats = intrinsics_inv  # 存储投影矩阵（实际是 K 的 4x4 扩展）
    
    def set_img_names(self, img_names):
        """设置图像文件路径列表。"""
        self.img_names = img_names
            
    def add_all_frames(self, frames):
        """设置图像帧张量 (S, 3, H, W)。"""
        self.frames = frames
    
    def add_all_retrieval_vectors(self, retrieval_vectors):
        """设置图像检索向量（SALAD 特征向量，用于回环检测）。"""
        self.retrieval_vectors = retrieval_vectors
    
    def get_lc_status(self):
        """获取该子图是否为回环闭合子图。"""
        return self.is_lc_submap
    
    def get_id(self):
        """获取子图的全局唯一 ID。"""
        return self.submap_id

    def get_conf_threshold(self):
        """获取置信度过滤阈值。"""
        return self.conf_threshold
    
    def get_conf_masks_frame(self, index):
        """获取指定帧的置信度掩码。"""
        return self.conf_masks[index]
    
    def get_frame_at_index(self, index):
        """获取指定索引的图像帧张量 (3, H, W)。"""
        return self.frames[index, ...]
    
    def get_last_non_loop_frame_index(self):
        """获取最后一个非回环帧的索引。"""
        return self.last_non_loop_frame_index
    
    def get_img_names_at_index(self, index):
        """获取指定索引的图像文件名。"""
        return self.img_names[index]

    def get_all_frames(self):
        """获取所有图像帧张量 (S, 3, H, W)。"""
        return self.frames
    
    def get_all_retrieval_vectors(self):
        """获取所有帧的图像检索向量。"""
        return self.retrieval_vectors

    def get_first_homography_world(self, graph):
        """
        获取子图第一帧在世界坐标系下的单应矩阵 (4x4)。
        从位姿图中查询，因此会包含后端优化的结果。

        Args:
            graph: PoseGraph 实例
        Returns:
            归一化后的 4x4 单应矩阵（最后元素为1）
        """
        homography =  graph.get_homography(self.get_id())
        homography = homography / homography[-1,-1]
        return homography

    def get_last_homography_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a 
        loop closure frame. 
        Returns a 4x4 matrix normalized so that the last element is 1.

        获取子图最后一个非回环帧在世界坐标系下的单应矩阵。
        """
        homography = graph.get_homography(self.get_id() + self.get_last_non_loop_frame_index())
        homography = homography / homography[-1,-1]
        return homography

    def get_first_pose_world(self, graph):
        """
        Get the first camera projection matrix of the submap. 
        Returns a 4x4 matrix normalized so that the last element is 1.

        获取子图第一帧在世界坐标系下的位姿（单应矩阵的逆）。
        """
        return np.linalg.inv(self.get_first_homography_world(graph))

    def get_last_pose_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a 
        loop closure frame. 
        Returns a 4x4 matrix normalized so that the last element is 1.

        获取子图最后一个非回环帧在世界坐标系下的投影矩阵。
        """
        projection_mat = graph.get_projection_matrix(self.get_id() + self.get_last_non_loop_frame_index())
        projection_mat = projection_mat / projection_mat[-1,-1]
        return projection_mat

    def get_all_poses(self):
        """获取子图中所有帧的位姿（世界到相机变换矩阵）。"""
        return self.poses

    def get_all_poses_world(self, graph, give_camera_mat=False):
        """
        获取子图中所有帧在世界坐标系下的位姿。

        通过位姿图中存储的单应矩阵和内参矩阵计算每帧的世界位姿。
        这些位姿会反映后端优化和回环修正的结果。

        Args:
            graph: PoseGraph 实例
            give_camera_mat: 如果为 True，返回完整的投影矩阵 (K @ inv(H))；
                            否则返回分解后的 SE(3) 位姿

        Returns:
            所有帧的世界位姿 (N, 4, 4)
        """
        # 从位姿图中查询每帧的单应矩阵
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        poses = []
        for index, homography_world in enumerate(homography_list):
            # 投影矩阵 = K @ inv(H_world)
            projection_mat = self.proj_mats[index] @ np.linalg.inv(homography_world) # TODO HERE
            projection_mat = projection_mat / projection_mat[-1,-1]
            if give_camera_mat:
                poses.append(projection_mat)
            else:
                # 分解投影矩阵为内参 K、旋转 R、平移 t
                cal, rot, trans, scale = decompose_camera(projection_mat[0:3,:])

                pose = np.eye(4)
                pose[0:3, 0:3] = rot
                pose[0:3,3] = trans
                poses.append(pose)
        return np.stack(poses, axis=0)
    
    def get_frame_pointcloud(self, pose_index):
        """获取指定帧的3D点云 (H, W, 3)。"""
        return self.pointclouds[pose_index]

    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names, 
        removing any leading zeros, and add them all to a list.

        从文件名中提取帧编号（整数或小数），去除前导零。
        例如 "frame_0042.png" → 42.0

        Note: This does not include any of the loop closure frames.
        注意：不包含任何回环闭合帧。
        """
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        """设置最后一个非回环帧的索引。"""
        self.last_non_loop_frame_index = last_non_loop_frame_index
    
    def set_all_retrieval_vectors(self, retrieval_vectors):
        """设置所有帧的图像检索向量。"""
        self.retrieval_vectors = retrieval_vectors
    
    def set_conf_masks(self, conf_masks):
        """设置所有帧的置信度掩码。"""
        self.conf_masks = conf_masks
    
    def set_all_semantic_vectors(self, semantic_vectors):
        """设置所有帧的 CLIP 语义嵌入向量。"""
        self.semantic_vectors = semantic_vectors

    def get_pose_subframe(self, pose_index):
        """获取指定帧的相机到世界变换（位姿的逆）。"""
        return np.linalg.inv(self.poses[pose_index])
    
    def get_frame_ids(self):
        """
        获取所有帧的帧编号列表。
        # Note this does not include any of the loop closure frames
        注意：不包含任何回环帧。
        """
        return self.frame_ids

    def filter_data_by_confidence(self, data):
        """
        根据置信度阈值过滤数据。

        使用 self.conf > self.conf_threshold 生成布尔掩码，
        仅保留置信度高于阈值的元素。

        Args:
            data: 与 self.conf 形状一致的数据（如点云或颜色）

        Returns:
            过滤后的数据（1D 或扁平化）
        """
        init_conf_mask = self.conf > self.conf_threshold
        return data[init_conf_mask]

    def get_points_list_in_world_frame(self, graph, rectifing_homographies=None):
        """
        获取每帧的3D点云列表（已变换到世界坐标系）。

        与 get_points_in_world_frame 不同，此方法返回按帧分开的列表，
        不会将所有帧的点合并。同时返回帧编号和置信度掩码。

        Args:
            graph: PoseGraph 实例
            rectifing_homographies: 可选的校正单应矩阵（暂未使用）

        Returns:
            point_list: 每帧的世界坐标系点云列表
            frame_id_list: 帧编号列表
            frame_conf_mask: 每帧的置信度掩码列表
        """
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            # 将3D点转为齐次坐标并应用单应矩阵变换到世界坐标系
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            # 齐次坐标归一化
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            point_list.append(points_transformed)
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] > self.conf_threshold
            frame_conf_mask.append(conf_mask)
        return point_list, frame_id_list, frame_conf_mask

    def get_points_in_world_frame(self, graph):
        """
        获取子图所有帧的3D点云，合并为一个数组（世界坐标系）。

        仅保留置信度高于阈值的点，所有帧的点合并在一起。
        用于可视化时显示整个子图的点云。

        Args:
            graph: PoseGraph 实例

        Returns:
            合并后的世界坐标系点云 (M, 3)，M 为过滤后的总点数
        """
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        points_all = None
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            # 齐次坐标变换
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            # 根据置信度掩码过滤低质量点
            conf_mask = self.conf_masks[index] > self.conf_threshold

            points_transformed = points_transformed[conf_mask]
            if index == 0:
                points_all = points_transformed
            else:
                points_all = np.vstack([points_all, points_transformed])

        return points_all

    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        """
        获取体素降采样后的点云（世界坐标系）。

        使用 Open3D 的体素降采样和半径离群点移除来生成
        更紧凑的点云表示，适用于高效的可视化和存储。

        Args:
            voxel_size: 体素大小（单位：米）
            nb_points: 半径内最少点数（用于离群点移除）
            factor_for_outlier_rejection: 离群点判断半径 = voxel_size * factor

        Returns:
            体素化后的 Open3D 点云对象（世界坐标系）
        """
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                # 先按置信度过滤
                points = self.filter_data_by_confidence(self.pointclouds)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_confidence(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                # 创建 Open3D 点云并进行体素降采样
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                # 移除离群点（半径内点数不足的点）
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        # 将体素化的点云变换到世界坐标系
        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        H_world_map = self.poses[0]  # 使用第一帧的位姿作为变换
        points_transformed = (H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame
    
    def get_points_colors(self):
        """
        获取置信度过滤后的点云颜色。
        
        Returns:
            颜色数组 (M, 3)，值域 [0, 255]
        """
        colors = self.filter_data_by_confidence(self.colors)
        return colors.reshape(-1, 3)

    def get_all_semantic_vectors(self):
        """获取所有帧的 CLIP 语义嵌入向量。"""
        return self.semantic_vectors

    
    def get_points_in_mask(self, frame_index, mask, graph):
        """
        获取指定帧中被掩码选中的3D点（世界坐标系）。

        用于开放集语义搜索：SAM3 分割出的掩码区域对应的3D点，
        可用于计算有向包围盒（OBB）。

        Args:
            frame_index: 帧索引
            mask: 2D 掩码 (H, W)，布尔类型
            graph: PoseGraph 实例

        Returns:
            掩码内的3D点 (K, 3)
        """
        points = self.get_points_list_in_world_frame(graph)[0][frame_index]
        points_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        points_in_mask = points_flat[mask_flat]
        return points_in_mask
