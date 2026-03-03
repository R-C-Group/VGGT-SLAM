"""
frame_overlap.py — 光流帧跟踪器（关键帧选择）模块

本模块实现了基于光流（Optical Flow）的关键帧选择策略。

原理：
- 使用 Lucas-Kanade 光流跟踪特征点
- 计算当前帧与上一关键帧之间特征点的平均位移量（视差）
- 当平均视差超过阈值时，认为两帧之间有足够的运动，将当前帧选为新关键帧
- 这种方法可以自适应地根据相机运动速度调整关键帧密度

优点：
- 比固定间隔采样更灵活，适应不同的运动速度
- 当相机静止时不会生成冗余帧
- 当相机快速运动时能及时捕获关键帧
"""

import argparse
import torch
import numpy as np
import cv2

class FrameTracker:
    """
    光流帧跟踪器。

    通过计算连续帧之间的光流视差来决定
    是否选择当前帧作为新的关键帧。

    Attributes:
        last_kf: 上一个关键帧的原始图像（BGR）
        kf_pts: 上一个关键帧中检测到的角点 (N, 1, 2)
        kf_gray: 上一个关键帧的灰度图
    """
    def __init__(self):
        self.last_kf = None    # 上一关键帧图像
        self.kf_pts = None     # 关键帧中的特征点
        self.kf_gray = None    # 关键帧灰度图

    def initialize_keyframe(self, image):
        """
        初始化新的关键帧。

        使用 Shi-Tomasi 角点检测器（goodFeaturesToTrack）
        在当前图像中检测特征点。

        Args:
            image: BGR 格式的图像（OpenCV 格式）
        """
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Shi-Tomasi 角点检测
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray,
            maxCorners=1000,      # 最多检测 1000 个角点
            qualityLevel=0.01,    # 质量阈值（相对于最佳角点）
            minDistance=8,        # 角点之间的最小距离
            blockSize=7           # 角点检测的邻域大小
        )

    def compute_disparity(self, image, min_disparity, visualize=False):
        """
        计算当前帧与上一关键帧之间的光流视差。

        使用 Lucas-Kanade 金字塔光流法跟踪关键帧中的特征点到当前帧，
        计算跟踪成功的特征点的平均位移量。

        如果平均位移量超过 min_disparity，则认为有足够的运动，
        将当前帧初始化为新关键帧。

        Args:
            image: 当前帧图像（BGR 格式）
            min_disparity: 最小视差阈值（像素），超过则选为新关键帧
            visualize: 是否可视化光流箭头

        Returns:
            True: 当前帧已被选为新关键帧（视差足够大或首帧）
            False: 当前帧与关键帧太相似，跳过
        """
        # 如果没有上一关键帧，或者特征点不足，初始化当前帧为关键帧
        if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < 10:
            self.initialize_keyframe(image)
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame
        # 使用 Lucas-Kanade 金字塔光流跟踪关键帧的特征点到当前帧
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21),      # 搜索窗口大小
            maxLevel=3,            # 金字塔层数
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # 终止条件
        )

        # 筛选跟踪成功的点
        status = status.flatten()
        good_kf = self.kf_pts[status == 1]      # 关键帧中跟踪成功的点
        good_next = next_pts[status == 1]         # 当前帧中对应的点

        # 如果跟踪成功的点太少，重新初始化
        if len(good_kf) < 10:
            self.initialize_keyframe(image)
            return True

        # Measure displacement from keyframe to current frame
        # 计算特征点位移的 L2 范数
        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disparity = np.mean(displacement)  # 平均视差（像素）

        # 可选：可视化光流箭头
        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_kf, good_next):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)

        # 判断视差是否超过阈值
        if mean_disparity > min_disparity:
            # 视差足够大，选为新关键帧
            self.initialize_keyframe(image)
            return True
        else:
            # 视差太小，跳过当前帧
            return False