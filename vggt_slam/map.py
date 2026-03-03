"""
map.py — 全局地图（GraphMap）管理模块

本模块实现了 GraphMap 类，负责管理所有子图（Submap）的集合。

主要功能：
1. 子图的添加和检索
2. 基于 CLIP 语义嵌入的帧检索（开放集语义搜索）
3. 基于 SALAD 向量的帧检索（回环检测）
4. 位姿的导出（TUM 格式或 KITTI 格式）
5. 点云的导出（PCD 格式或逐帧 NPZ 格式）
"""

import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R  # 旋转矩阵与四元数互转
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import decompose_camera, cosine_similarity

class GraphMap:
    """
    全局地图管理器，存储和管理所有子图。

    Attributes:
        submaps: 子图字典 {submap_id: Submap}
        rectifying_H_mats: 校正单应矩阵列表（暂用于逐帧点云保存时的断言校验）
        non_lc_submap_ids: 非回环子图的 ID 列表（按添加顺序）
    """
    def __init__(self):
        self.submaps = dict()            # {submap_id: Submap} 子图字典
        self.rectifying_H_mats = []      # 校正单应矩阵列表
        self.non_lc_submap_ids = []      # 非回环子图 ID 列表（有序）
    
    def get_num_submaps(self):
        """获取地图中子图的总数量。"""
        return len(self.submaps)

    def add_submap(self, submap):
        """
        将子图添加到全局地图中。

        Args:
            submap: Submap 实例
        """
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        # 记录非回环子图的 ID（有序列表，用于跳过最近的子图以避免误检测回环）
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        获取所有子图中最大的 ID 值。

        Args:
            ignore_loop_closure_submaps: 是否忽略回环子图

        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        """根据 ID 获取子图。"""
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        """获取最新添加的子图（ID 最大的子图）。"""
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        """
        基于 CLIP 语义向量检索与查询文本最匹配的帧。

        遍历所有非回环子图的所有帧，计算每帧的图像语义嵌入
        与查询文本嵌入之间的余弦相似度，返回得分最高的帧。

        用于开放集语义搜索功能（--run_os）。

        Args:
            query_text_vector: 文本查询的 CLIP 嵌入向量

        Returns:
            (best_score, best_submap_id, best_frame_index): 最佳匹配信息
        """
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue  # 跳过回环子图
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                # 计算余弦相似度
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        """
        基于 SALAD 检索向量查找与查询向量最相似的帧（用于回环检测）。

        使用 L2 范数距离（越小越相似）而非余弦相似度。
        会跳过当前子图和最近的子图（避免相邻帧被误检测为回环）。

        Args:
            query_vector: 查询帧的 SALAD 检索向量
            current_submap_id: 当前子图 ID（需跳过）
            ignore_last_submap: 是否跳过最近添加的非回环子图

        Returns:
            (best_score, best_submap_id, best_frame_index): 最佳匹配信息
            注意：这里的 score 是 L2 距离，越小越好
        """
        overall_best_score = 1000  # 初始化为大值（L2 距离越小越好）
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            # 跳过当前子图
            if submap_key == current_submap_id:
                continue

            # 跳过最近的非回环子图（防止相邻子图误触发回环）
            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue  # 跳过回环子图
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    # 计算 L2 范数距离
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                # 当前仅支持每个子图最多一个回环
                
                best_score_id = np.argmin(scores)  # 取 L2 距离最小的
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        """
        从检测到的回环中提取被检索到的历史帧。

        Args:
            loops: LoopMatch 列表

        Returns:
            frames: 检索到的帧张量列表
        """
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        """获取所有子图的值集合。"""
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        """按 ID 排序的子图生成器。"""
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        """
        获取所有子图中所有帧的优化后单应矩阵。

        Args:
            graph: PoseGraph 实例

        Returns:
            所有单应矩阵堆叠的 numpy 数组
        """
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        """
        获取所有非回环子图中所有帧的世界位姿。

        Args:
            graph: PoseGraph 实例
            give_camera_mat: 是否返回完整投影矩阵

        Returns:
            所有位姿堆叠的 numpy 数组
        """
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue  # 跳过回环子图
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        """
        将所有位姿保存到文件。

        支持两种格式：
        - TUM 格式（默认）：timestamp tx ty tz qx qy qz qw
        - KITTI 格式：timestamp + 3x4 位姿矩阵展平（12个值）

        Args:
            file_name: 输出文件路径
            graph: PoseGraph 实例
            give_camera_mat: 是否使用投影矩阵
            kitti_format: 是否使用 KITTI 格式
        """
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue  # 跳过回环子图
                frame_ids = submap.get_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    # 分解投影矩阵为内参 K、旋转 R 和平移 t
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    if kitti_format:
                        # KITTI 格式：3x4 位姿矩阵展平
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]  # 去掉最后一行 [0,0,0,1]
                        output = np.array([float(frame_id), *output])
                    else:
                        # TUM 格式：将旋转矩阵转为四元数
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, graph, file_name):
        """
        逐帧保存稠密点云到指定目录。

        每帧保存为一个 .npz 文件，包含：
        - pointcloud: 3D点云 (H, W, 3)
        - mask: 置信度掩码 (H, W)

        Args:
            graph: PoseGraph 实例
            file_name: 输出目录路径
        """
        os.makedirs(file_name, exist_ok=True)
        count = 0
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
                count += len(submap.poses)  # 注意：此行因 continue 不会执行（可能是 bug）
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
        assert count == len(self.rectifying_H_mats), "Number of rectifying mats and number of point maps do not match"
                

    def write_points_to_file(self, graph, file_name):
        """
        将所有子图的点云合并并保存为 PCD 文件。

        Args:
            graph: PoseGraph 实例
            file_name: 输出 PCD 文件路径
        """
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(graph)
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0  # 归一化颜色到 [0, 1]
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)
