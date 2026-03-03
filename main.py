"""
VGGT-SLAM 主入口文件

本文件实现了 VGGT-SLAM 的完整流水线，包括：
1. 加载 VGGT 前馈视觉模型
2. 读取输入图像序列并进行关键帧选择（基于光流视差）
3. 对关键帧集合构建子图（Submap）并进行 VGGT 推理
4. 构建位姿图并在 SL(4) 流形上进行后端优化
5. 检测回环闭合并修正全局一致性
6. 可选的开放集语义搜索（CLIP + SAM3 → 3D OBB）
7. 结果可视化（Viser 3D 查看器）和日志保存
"""

import os
import glob
import time
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image  # 将张量转换为 PIL 图像
from tqdm.auto import tqdm  # 进度条
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils          # SLAM 工具函数集合（排序、降采样、分解、计时等）
from vggt_slam.solver import Solver            # 核心求解器：管理模型推理、子图、位姿图、回环、可视化
from vggt_slam.submap import Submap            # 子图（Submap）数据结构

from vggt.models.vggt import VGGT             # VGGT 前馈视觉几何模型（第三方）

# ==================== 命令行参数定义 ====================
parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
# --vis_map: 是否在构建过程中实时可视化点云（否则仅在结束后展示最终地图）
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
# --vis_voxel_size: 可视化降采样体素大小（例如 0.05 表示 5cm），None 表示不降采样
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
# --run_os: 启用开放集语义搜索（需要 Perception Encoder CLIP 和 SAM3）
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
# --vis_flow: 可视化光流以辅助关键帧选择调试
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
# --submap_size: 每个子图中的新帧数量（不含重叠帧和回环帧）
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
# --overlapping_window_size: 重叠窗口大小，用于子图间 SL(4) 尺度对齐（目前仅支持 1）
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
# --max_loops: 每个子图最大回环检测数量（仅支持 0 或 1）
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
# --min_disparity: 触发新关键帧的最小光流平均视差（像素）
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
# --conf_threshold: 置信度过滤百分位数（如 25 表示过滤最低 25% 的低置信度点）
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
# --lc_thres: 图像检索相似度阈值。范围 [0,1.0]，越高则回环检测越敏感
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    主函数：封装 VGGT-SLAM 完整流水线。
    流程：图像加载 → 光流关键帧选择 → 子图推理 → 位姿图构建+优化 → 回环检测 → 可视化/保存
    """
    args = parser.parse_args()

    use_optical_flow_downsample = True  # 是否启用光流自适应关键帧选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化求解器：管理模型推理、子图、位姿图、回环检测和可视化
    solver = Solver(
        init_conf_threshold=args.conf_threshold,  # 置信度过滤百分位数
        lc_thres=args.lc_thres,                    # 回环检测相似度阈值
        vis_voxel_size=args.vis_voxel_size          # 可视化降采样体素大小
    )

    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()  # 构建 SAM3 图像分割模型
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        # 加载 CLIP 模型，用于图像-文本语义匹配
        # clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
        clip_model.load_ckpt("/home/kwanwaipang/.cache/torch/hub/PE-Core-L14-336.pt")
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)  # 文本分词器
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)    # 图像预处理变换
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    # ==================== 加载 VGGT 前馈视觉几何模型 ====================
    # VGGT (Visual Geometry Grounded Transformer): 同时预测多帧深度、位姿、3D点和置信度
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()                         # 设置为评估模式（关闭 dropout 等）
    model = model.to(torch.bfloat16)     # 使用 bfloat16 半精度以节省显存
    model = model.to(device)

    # ==================== 加载和排序图像 ====================
    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    # 过滤掉深度图、txt 文件和 db 文件，只保留 RGB 图像
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)  # 按文件名数字排序（确保时间顺序）
    downsample_factor = 1  # 均匀降采样因子（1 = 保留所有图像）
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    # ==================== 主循环：逐帧处理 ====================
    image_names_subset = []      # 当前子图待处理的关键帧列表
    count = 0                     # 已处理的子图数量
    image_count = 0               # 已处理的关键帧数量
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()   # 关键帧选择耗时计时器
    backend_time = utils.Accumulator()    # 后端优化耗时计时器
    for image_name in tqdm(image_names):
        # ---------- 关键帧选择（基于光流视差判断）----------
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                # 计算当前帧与上一关键帧的光流视差，若超过阈值则作为新关键帧
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # ---------- 子图处理：收集到足够关键帧或到达最后一帧时触发 ----------
        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            # 运行 VGGT 模型推理 + 回环检测，获取深度、位姿、3D点等预测结果
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            # 将预测的3D点和位姿添加到全局地图和位姿图中
            solver.add_points(predictions)

            # 后端优化：使用 GTSAM LM 优化器在 SL(4) 流形上优化全局位姿图
            with backend_time:
                solver.graph.optimize()

            # 检测到回环时需要更新所有子图可视化（因为位姿已被修正）
            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()     # 回环修正后更新全部子图
                else:
                    solver.update_latest_submap_vis()  # 仅更新最新子图
            
            # Reset for next submap.
            # 保留最后 overlapping_window_size 帧作为下一子图的起始重叠帧（用于尺度对齐）
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    # ==================== 输出统计信息 ====================
    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    # ==================== 开放集语义搜索交互循环 ====================
    # 用户输入文本查询 → CLIP 检索最匹配帧 → SAM3 分割目标 → 3D OBB 可视化
    if args.run_os:
        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            start_time = time.time()
            # 计算文本查询的 CLIP 嵌入向量
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            # 在所有子图的所有帧中搜索语义最匹配的帧
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                # 用 SAM3 对最佳匹配帧进行文本驱动的实例分割
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())

            print("Time taken for query:", time.time() - start_time)

            # 在图像上叠加分割掩码并显示
            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            # 对每个检测到的掩码，从3D点云中提取对应点，计算 OBB 并在 Viser 中绘制
            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    # ==================== 最终可视化与日志保存 ====================
    if not args.vis_map:
        # just show the map after all submaps have been processed
        # 未在过程中实时可视化，则最后一次性展示完整地图
        solver.update_all_submap_vis()

    if args.log_results:
        # 保存位姿到文件（TUM 格式：timestamp tx ty tz qx qy qz qw）
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            # 保存每帧的稠密点云（包含点云和置信度掩码）
            solver.map.save_framewise_pointclouds(solver.graph, args.log_path.replace(".txt", "_logs"))


if __name__ == "__main__":
    main()
