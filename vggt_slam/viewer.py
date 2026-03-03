"""
viewer.py — Viser 3D 可视化查看器模块

本模块实现了基于 Viser 的 3D 可视化功能。
Viser 是一个基于 Web 的 3D 可视化服务器，支持实时查看和交互。

主要功能：
1. 显示相机帧（坐标轴 + 视锥体 + 图像缩略图）
2. 显示有向包围盒（OBB）线框
3. 支持 "Play Walkthrough" 按钮实现相机漫游动画
4. 支持 "Show Cameras" 复选框控制相机的显示/隐藏

使用方式：
- 启动服务后，在浏览器中访问 http://localhost:8080/
- 不同子图的相机用不同颜色标识
"""

import time
from typing import Dict, List
import numpy as np
import torch
import viser                      # Viser：基于 Web 的 3D 可视化库
import viser.transforms as viser_tf  # Viser 的变换工具（SE3 等）


class Viewer:
    """
    Viser 3D 可视化查看器。

    启动一个 Web 服务器，在浏览器中实时展示 3D 点云、
    相机位姿和有向包围盒。

    Attributes:
        server: Viser 服务器实例
        submap_frames: 每个子图的坐标帧句柄字典 {submap_id: [FrameHandle]}
        submap_frustums: 每个子图的视锥体句柄字典 {submap_id: [FrustumHandle]}
        random_colors: 随机颜色数组（用于区分不同子图）
    """
    def __init__(self, port: int = 8080):
        """
        初始化 Viser 3D 可视化服务器。

        Args:
            port: 服务器端口号（默认 8080，浏览器访问 http://localhost:8080/）
        """
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- GUI Elements ---
        # 「显示相机」复选框
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Add a button to trigger the walkthrough
        # 「播放漫游」按钮：按下后自动遍历所有相机位姿
        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        # 存储每个子图的可视化句柄
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}       # 坐标帧
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}  # 视锥体

        # 为不同子图分配不同的随机颜色
        num_rand_colors = 250
        np.random.seed(100)  # 固定随机种子以保证颜色一致性
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()  # 子图 ID → 颜色索引的映射
        self.obj_id = 0  # OBB 对象的自增 ID

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        为指定子图添加相机帧和视锥体到3D场景中。

        每个相机帧包括：
        - 坐标轴（xyz三色轴）
        - 视锥体（带图像缩略图的相机锥体）

        Args:
            extrinsics: 相机到世界的变换矩阵 (S, 3, 4)
            images_: 图像帧张量 (S, 3, H, W)
            submap_id: 子图 ID（用于颜色区分和命名）
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        # 为新子图分配颜色
        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]  # 帧数
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            # 将 3x4 变换矩阵转为 SE3 对象
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            # 添加坐标轴（表示相机的位置和朝向）
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,    # 坐标轴长度
                axes_radius=0.002,   # 坐标轴粗细
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            # 添加视锥体（带图像缩略图）
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)  # (C,H,W) → (H,W,C)
            h, w = img.shape[:2]
            fy = 1.1 * h        # 估计的焦距（用于可视化）
            fov = 2 * np.arctan2(h / 2, fy)  # 垂直视场角

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,    # 宽高比
                scale=0.05,       # 视锥体大小
                image=img,        # 缩略图
                line_width=3.0,   # 线宽
                color=self.random_colors[self.submap_id_to_color[submap_id]]  # 子图对应的颜色
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps.
        切换所有相机帧和视锥体的可见性。"""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def visualize_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color = (255, 0, 0),
        line_width: float = 2.0,
    ):
        """
        Visualize an oriented bounding box (OBB) in Viser.
        在 Viser 场景中可视化有向包围盒（OBB）。

        OBB 以12条线段的线框形式绘制。

        Parameters
        ----------
        name : str
            Identifier for the OBB in the scene.  OBB 在场景中的标识
        center : (3,) array
            World-space center of the OBB.  OBB 的世界坐标系中心
        extent : (3,) array
            Full side lengths of the OBB (dx, dy, dz).  OBB 的总边长
        rotation : (3,3) array
            Rotation matrix of the OBB in world coordinates.  OBB 的旋转矩阵
        color : tuple[int,int,int]
            RGB color of the wireframe box.  线框颜色
        line_width : float
            Thickness of the box edges.  线条粗细

        Notes
        -----
        The box is drawn as a wireframe with 12 edges.
        包围盒以12条棱线的线框形式绘制。
        """

        # Compute local corners (8)
        # 计算局部坐标系下的8个顶点
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float32)  # shape (8,3)

        # Transform to world
        # 通过旋转和平移将顶点变换到世界坐标系
        corners_world = (rotation @ corners_local.T).T + center  # shape (8,3)

        # Build edges (12 line segments) as start/end pairs
        # 构建12条棱线（底面4条 + 顶面4条 + 竖直4条）
        edges_idx = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face 底面
            (4,5),(5,6),(6,7),(7,4),  # top face    顶面
            (0,4),(1,5),(2,6),(3,7)   # vertical edges 竖直边
        ]

        segments = []
        for (i,j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        # segments is list of length 24, reshape into (N,2,3)
        # 将线段列表整形为 (12, 2, 3) 的数组
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        name = f"obb_{self.obj_id}"
        self.obj_id += 1
        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=color,       # single color for all segments 所有线段使用相同颜色
            line_width=line_width,
            visible=True
        )

    def run_walkthrough(self, fps: float = 20.0):
            """
            Walks through the map using the current live positions of all frames.
            This accounts for loop closures because it pulls data from the scene handles.

            使用所有帧的当前位姿进行相机漫游动画。
            由于直接从场景句柄中读取位姿，因此会自动反映回环修正后的结果。

            Args:
                fps: 漫游帧率（帧/秒）
            """
            # 1. Gather all submap IDs and sort them to ensure a logical sequence
            # 1. 收集所有子图 ID 并排序
            sorted_submap_ids = sorted(self.submap_frames.keys())
            
            if not sorted_submap_ids:
                print("No frames found to walk through.")
                return

            clients = self.server.get_clients()
            if not clients:
                print("No clients connected to perform walkthrough.")
                return

            print("Starting walkthrough of updated poses...")

            # 2. 按子图顺序遍历每一帧，更新客户端相机位置
            for sub_id in sorted_submap_ids:
                frames = self.submap_frames[sub_id]
                # Assumes frames were added in chronological order to the list
                for frame_handle in frames:
                    # Get the current world-space pose from the visualizer
                    # If a loop closure moved the submap, these values will be updated
                    # 从可视化场景中获取当前帧的世界位姿
                    current_pos = frame_handle.position
                    current_wxyz = frame_handle.wxyz

                    # Update all connected clients
                    # 更新所有已连接客户端的相机视角
                    for client in clients.values():
                        client.camera.position = current_pos
                        client.camera.wxyz = current_wxyz
                    
                    # Control speed (1/fps)
                    # 控制漫游速度
                    time.sleep(1.0 / fps)