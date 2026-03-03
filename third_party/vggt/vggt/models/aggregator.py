# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from matplotlib import pyplot as plt

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

viz_attention = False

def plot_first_token_cosine_matrix(token1, token2, eps=1e-8):
    """
    token1, token2: np arrays of shape (n, 782, 1024)
    Uses only token index 0 from each.
    """

    # Normalize

    # Extract first tokens (shape: n x 1024)
    t1 = np.mean(token1[:, 5:, :], axis=1)  # (n x 1024)
    t2 = np.mean(token2[:, 5:, :], axis=1)  # (n x 1024)

    t1 = t1 / (np.linalg.norm(t1, axis=-1, keepdims=True) + eps)
    t2 = t2 / (np.linalg.norm(t2, axis=-1, keepdims=True) + eps)

    # Compute cosine similarity: (t1 · t2^T)
    sim = t1 @ t2.T  # (n x n) matrix

    # Plot the matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(sim, interpolation="nearest")
    plt.colorbar(label="cosine similarity")
    plt.title("Cosine Similarity Matrix (First Token)")
    plt.xlabel("Image index")
    plt.ylabel("Image index")
    plt.tight_layout()
    plt.show()

    return sim

# from image_retrieval import run_asmk

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

import torch
import matplotlib.pyplot as plt

def mean_top_quarter(arr):
    """
    计算数组中前 25% 最大值的均值。
    对应论文 §IV-B 公式6：取所有比率的 top 25% 的均值作为最终匹配分数。
    """
    # flatten to 1D
    flat = arr.ravel()
    # find 75th percentile threshold  找到 75% 分位数阈值
    thresh = np.percentile(flat, 75)
    # select values >= threshold  选择 >= 阈值的值（即 top 25%）
    top_vals = flat[flat >= thresh]
    # return their mean  返回其均值
    return top_vals.mean()


# k_selected = k[:, :, token_idx+frame_idx*tokens_per_img:token_idx+frame_idx*tokens_per_img+1, :] 
# attn = q @ k_selected.transpose(-2, -1)
# attn = attn.transpose(-2, -1) 
# attn = attn.softmax(dim=-1)    
# attn_mean = attn.mean(dim=1)
# attn_mean = attn_mean.view(B, N)
# shown = attn_mean[:, self.token_offset:tokens_per_img]

def get_similarity(k, q, token_offset=5, image_height=21, image_width=37):
    """
    计算两张图像之间的 VGGT Layer 22 匹配分数 α_match。
    
    这是论文 §IV-B 的核心实现：
    1. 从 Layer 22 提取 key (k) 和 query (q) tokens
    2. 计算图像2的 query tokens 对图像1的 key tokens 的注意力
    3. 除以图像1内部的最大自注意力值（归一化）
    4. 取 top 25% 比率的均值作为最终匹配分数
    
    Args:
        k: key tokens (B, num_heads, N, head_dim)
        q: query tokens (B, num_heads, N, head_dim)
        token_offset: 特殊 token 偏移量（camera + register tokens）
        
    Returns:
        ratio: α_match 匹配分数（值越高表示两帧重叠越大，> 0.85 确认回环）
    """
    num_imgs = 2
    tokens_per_img = q.shape[2] // num_imgs
    # 只使用图像1的 key tokens（跳过特殊 tokens）
    k = k[:,:,token_offset:tokens_per_img , :]

    # 计算注意力分数: Q @ K^T
    attn = q @ k.transpose(-2, -1)
    attn = attn.transpose(-2, -1)
    attn = attn.softmax(dim=-1)  # softmax 归一化
    attn = attn.mean(dim=1)  # 对所有注意力头取均值

    # 分离两张图像的注意力分数
    all_token_to_first_frame = attn[..., :tokens_per_img]    # 所有 token 对图像1的注意力
    all_token_to_second_frame = attn[..., tokens_per_img:]   # 所有 token 对图像2的注意力

    # 论文公式5：计算每个 key token 在图像1中获得的最大注意力值
    max_per_token_first_img = all_token_to_first_frame.max(dim=-1)[0]

    # 论文公式5-6：用图像1的最大注意力归一化图像2的注意力
    # 直觉：如果图像2中也有高注意力区域，说明 VGGT 找到了跨帧对应关系
    attn_second_frame_normalized = all_token_to_second_frame / (max_per_token_first_img.unsqueeze(-1) + 1e-8)
    ratio = attn_second_frame_normalized.max(dim=1)[0]

    ratio =  ratio.float().detach().cpu().numpy()
    # 论文公式6：取 top 25% 的均值作为最终匹配分数
    ratio = mean_top_quarter(ratio)

    print("Average of top quarter attention values (all frames):", ratio)

    return ratio

class CosineAttentionVisualizer:
    def __init__(self, image_height=21, token_offset=5):
        self.image_width = 37
        self.image_height= image_height
        self.token_offset = token_offset  # number of special tokens to skip (e.g., CLS, etc.)
        self.layers = []  # each: {"name": str, "x_norm": tensor[B,N,C], "attn": tensor[B,N,N] or None}
        self.current_layer = 0
        self.mode = "cosine"  # "cosine" | "attention"
        self.last_clicked = None  # (frame_idx, row, col, token_idx)
        self.fig = None
        self.axes = []
        self.ims = []
        self.colorbar = None

    def add_layer(self, name, x, k, q):
        """
        x: (B, N, C) token embeddings (torch Tensor, CPU or CUDA)
        attn: (B, H, N, N) or None. If given, heads will be averaged to (B, N, N).
        """
        eps = 1e-8
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)

        self.layers.append({"name": name, "x_norm": x_norm, "k": k, "q": q})

    # ---------- plotting & interaction ----------

    def show(self):
        if not self.layers:
            print("No layers added.")
            return

        # Default reference: center-ish pixel in frame 0
        if self.last_clicked is None:
            r = min(self.image_height // 2, self.image_width - 1)
            c = min(self.image_width // 2, self.image_width - 1)
            token_idx = self.token_offset + r * self.image_width + c
            self.last_clicked = (0, r, c, token_idx)

        self._init_figure()
        self._redraw()
        plt.show()

    def _init_figure(self):
        layer = self.layers[self.current_layer]
        B = layer["x_norm"].shape[0]

        frames_per_row = 5
        ncols = min(B, frames_per_row)
        nrows = (B + ncols - 1) // ncols

        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 4.5*nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        self.axes = [ax for row in axes for ax in row]
        self.ims = []

        # connect events once
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _compute_maps(self):
        """Return (maps: B x (W*W), vmin, vmax) for current mode using current ref token."""
        layer = self.layers[self.current_layer]
        x_norm = layer["x_norm"]      # (B, N, C)
        # x_norm = torch.tensor(np.load('/home/dominic/vggt/tokens1.npy'))
        k = layer["k"]
        q = layer["q"]
        print(self.current_layer, k.shape, q.shape, x_norm.shape)
        B, N, C = x_norm.shape

        frame_idx, row, col, token_idx = self.last_clicked

        tokens_per_img = self.token_offset + self.image_height*self.image_width

        # guard if click outside valid token area
        if token_idx < self.token_offset or token_idx >= N:
            raise ValueError("Clicked token index out of range.")

        if self.mode == "cosine":
            # Use the *vector from the clicked frame* as the query, compare to ALL tokens in EVERY frame
            ref_vec = x_norm[frame_idx, token_idx, :]                     # (C,)
            # (C) x (B,N,C) -> (B,N)
            maps_full = torch.einsum("c,bnc->bn", ref_vec, x_norm)
            # display only the image tokens (skip special tokens)
            shown = maps_full[:, self.token_offset:tokens_per_img]
            # run_asmk(x_norm)
            print(self.current_layer, shown.shape)
            # np.save("/home/dominic/Documents/vggt/test_tokens/target_tokens12.npy", x_norm.detach().cpu().numpy())
            # plot_first_token_cosine_matrix(x_norm.detach().cpu().numpy(), x_norm.detach().cpu().numpy(), eps=1e-8)

            vmin, vmax = 0.0, 1.0
            return shown, vmin, vmax

        else:  # attention
            k_selected = k[:, :, token_idx+frame_idx*tokens_per_img:token_idx+frame_idx*tokens_per_img+1, :] 
            attn = q @ k_selected.transpose(-2, -1)
            print("here",k_selected.shape, q.shape, attn.shape)

            attn = attn.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)   
            print(attn.shape) 
            attn_mean = attn.mean(dim=1)
            attn_mean = attn_mean.view(B, N)
            shown = attn_mean[:, self.token_offset:tokens_per_img]
            # dynamic range across all shown frames for this selection
            vmin = float(shown.min().detach().cpu())
            vmax = float(shown.max().detach().cpu())
            # protect against degenerate ranges
            if vmin == vmax:
                vmax = vmin + 1e-8
            
            get_similarity(k, q)

            return shown, vmin, vmax

    def _redraw(self):
        maps, vmin, vmax = self._compute_maps()  # (B, W*W)
        B = maps.shape[0]
        W = self.image_width
        H = self.image_height

        # draw/update images
        if not self.ims:
            for i in range(len(self.axes)):
                ax = self.axes[i]
                if i < B:
                    frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                    im = ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Frame {i}")
                    # mark clicked pixel (same row/col) so you can see it
                    if i == self.last_clicked[0]:
                        ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2, color='black')
                    # else:
                    #     ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)
                    self.ims.append(im)
                ax.axis("off")

            # one shared colorbar
            self.colorbar = self.fig.colorbar(self.ims[0], ax=self.axes[:B],
                                              orientation="horizontal", fraction=0.03, pad=0.08)
        else:
            # update existing
            for i in range(B):
                frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                self.ims[i].set_data(frame)
                self.ims[i].set_clim(vmin, vmax)
                ax = self.axes[i]
                # clear old markers and replot marker
                # (remove all lines in this axes; cheap if there are only markers)
                for ln in list(ax.lines):
                    ln.remove()
                if i == self.last_clicked[0]:
                    ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2, color='black')
                # else:
                #     ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)

            if self.colorbar is not None:
                self.colorbar.update_normal(self.ims[0])

        self.fig.suptitle(
            f"Mode: {self.mode} • Layer: {self.layers[self.current_layer]['name']} • "
            f"Ref: frame={self.last_clicked[0]}, token_idx={self.last_clicked[3]} (row={self.last_clicked[1]}, col={self.last_clicked[2]})",
            fontsize=12
        )
        self.fig.canvas.draw_idle()

    # ---------- events ----------

    def _on_click(self, event):
        if event.inaxes not in self.axes:
            return
        ax_idx = self.axes.index(event.inaxes)

        # guard against clicks outside image
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        # map (row, col) -> token index
        token_idx = self.token_offset + row * self.image_width + col

        self.last_clicked = (ax_idx, row, col, token_idx)
        self._redraw()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self.current_layer = (self.current_layer + 1) % len(self.layers)
            self._redraw()
        elif event.key in ("left", "p"):
            self.current_layer = (self.current_layer - 1) % len(self.layers)
            self._redraw()
        elif event.key in ("m", "a"):  # toggle mode
            self.mode = "attention" if self.mode == "cosine" else "cosine"
            self._redraw()

class Aggregator(nn.Module):
    """
    VGGT 的核心骨干网络：交替注意力聚合器。
    
    基于 DINOv2-Large 视觉 Transformer，使用「帧内-全局」交替注意力（Alternating Attention）
    来学习多帧图像之间的几何关系。
    
    结构：24层，每层包含一个帧内注意力块 + 一个全局注意力块
    - frame_blocks: 帧内注意力（每帧独立处理，token shape = (B*S, P, C)）
    - global_blocks: 全局注意力（所有帧联合处理，token shape = (B, S*P, C)）
    
    特殊 tokens:
    - camera_token: 相机 token（2个位置：首帧 vs 其余帧）
    - register_token: 注册 token（4个，DINOv2 风格）
    
    关键参数：
        img_size: 图像大小（518px）
        patch_size: patch 大小（14，来自 DINOv2）
        embed_dim: 嵌入维度（1024，DINOv2-Large）
        depth: 层数（24）
        num_heads: 注意力头数（16）
        target_layer: 用于回环验证的层（默认 20=Layer 21，论文用 21=Layer 22）
        aa_order: 交替顺序 ["frame", "global"]，先帧内再全局
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        target_layer=20
    ):
        super().__init__()
        self.viz = CosineAttentionVisualizer()
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Rotary Position Embedding (RoPE) — 旋转位置编码,提供空间位置信息
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 目标层索引：用于回环验证的注意力层（论文§IV-B中的 Layer 22）
        self.target_layer = target_layer

        # 帧内注意力块（24层），每帧独立处理
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        # 全局注意力块（24层），所有帧联合处理
        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order  # 交替顺序：["frame", "global"]
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # 两种相机 token：首帧用 camera_token[0]，其余帧用 camera_token[1]
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        # patch token 起始索引 = 1(camera) + 4(register) = 5
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, compute_similarity=False) -> Tuple[List[torch.Tensor], int]:
        """
        Aggregator 的前向传播。
        
        处理流程：
        1. ImageNet 归一化 → Patch Embedding (DINOv2)
        2. 拼接 camera token + register token + patch tokens
        3. 交替执行 24 层: 帧内注意力 → 全局注意力
        4. 在 target_layer 时提取 target_tokens（用于图像检索）
        5. 如果 compute_similarity=True，在 target_layer 计算 α_match

        Args:
            images: 输入图像 [B, S, 3, H, W]
            compute_similarity: 是否计算 Layer 22 匹配分数（回环验证时使用）

        Returns:
            output_list: 各层的帧内+全局拼接特征列表
            patch_start_idx: patch token 起始索引（5）
            target_tokens: Layer 22 的归一化 tokens（用于 SALAD 检索）
            image_match_ratio: α_match 匹配分数（仅 compute_similarity=True 时有值）
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # global viz_attention
        # if compute_similarity:
        #     viz_attention = True

        # ImageNet 标准归一化
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        # 展平为 (B*S, C, H, W) 送入 DINOv2 提取 patch tokens
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)  # DINOv2 ViT patch embedding

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]  # 取归一化后的 patch tokens

        _, P, C = patch_tokens.shape  # P: 每帧的 patch 数量, C: 嵌入维度 (1024)

        # Expand camera and register tokens to match batch size and sequence length
        # 将特殊 tokens 扩展到 (B*S, X, C) 形状
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)   # 相机 token
        register_token = slice_expand_and_flatten(self.register_token, B, S)  # 注册 token

        # Concatenate special tokens with patch tokens
        # 拼接: [camera_token(1), register_tokens(4), patch_tokens(P)] → P+5 个 tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        target_tokens = None

        # eps = 1e-8
        # patch_tokens_save = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + eps)
        # np.save("/home/dominic/vggt/tokens2_decoder.npy", patch_tokens_save.detach().cpu().numpy())

        image_match_ratio  = None
        # ==================== 交替注意力循环（24层） ====================
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # 帧内注意力：tokens shape = (B*S, P, C)，每帧独立处理
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if compute_similarity and global_idx == self.target_layer:
                        # ★ 在目标层(Layer 22)计算匹配分数 α_match（论文§IV-B）
                        tokens, global_idx, global_intermediates, k, q = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, compute_similarity=True
                        )
                        # 调用 get_similarity 计算 α_match
                        image_match_ratio = get_similarity(k, q)
                    else:
                        # 全局注意力：tokens shape = (B, S*P, C)，跨帧联合处理
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    if global_idx == self.target_layer:
                        # ★ 在目标层提取 target_tokens（用于 SALAD 图像检索特征）
                        print(self.target_layer)
                        target_tokens = tokens.clone()
                        eps = 1e-8
                        # L2 归一化，使得余弦相似度计算更方便
                        target_tokens = target_tokens / (target_tokens.norm(dim=-1, keepdim=True) + eps)

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                # 拼接帧内和全局中间特征，维度翻倍 → 2*embed_dim
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        if viz_attention:
            self.viz.show()

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, target_tokens, image_match_ratio

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        处理帧内注意力块。
        tokens 保持 (B*S, P, C) 形状 — 每帧独立处理。
        帧内注意力学习每帧图像的局部特征（不涉及跨帧信息交换）。
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, viz_attention=False)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, compute_similarity=False):
        """
        处理全局注意力块。
        tokens 重塑为 (B, S*P, C) 形状 — 所有帧联合处理。
        全局注意力学习跨帧的几何关系（多帧之间的特征交互）。
        
        当 compute_similarity=True 时，额外返回 key 和 query tokens
        用于计算 Layer 22 匹配分数（论文§IV-B）。
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                if viz_attention or compute_similarity:
                    tokens, k, q = self.global_blocks[global_idx](tokens, pos=pos, viz_attention=True)
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos, viz_attention=False)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if viz_attention:
            # prior_tokens = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/" + str(global_idx) + ".npy"))
            # prior_tokens_k = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/k" + str(global_idx) + ".npy"))
            # prior_tokens_q = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/q" + str(global_idx) + ".npy"))
            # temp_concat = torch.cat([prior_tokens, tokens.detach().cpu()], dim=0)
            # temp_concat_k = torch.cat([prior_tokens_k, k.detach().cpu()], dim=2)
            # temp_concat_q = torch.cat([prior_tokens_q, q.detach().cpu()], dim=2)
            # np.save("/home/dominic/Documents/vggt/token_logs/" + str(global_idx) + ".npy", tokens.detach().cpu().numpy())
            # np.save("/home/dominic/Documents/vggt/token_logs/k" + str(global_idx) + ".npy", k.detach().cpu().numpy())
            # np.save("/home/dominic/Documents/vggt/token_logs/q" + str(global_idx) + ".npy", q.detach().cpu().numpy())
            # self.viz.add_layer("Layer " +str(global_idx), temp_concat, temp_concat_k, temp_concat_q)
            self.viz.add_layer("Layer " +str(global_idx), tokens, k, q)
        
        if compute_similarity:
            return tokens, global_idx, intermediates, k, q
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    处理特殊 tokens（camera/register），使首帧和其余帧使用不同的 token。
    
    输入 shape: (1, 2, X, C)
      - 位置0: 首帧专用 token
      - 位置1: 其余帧共用 token
    
    处理流程:
    1) 首帧使用 index=0 的 token
    2) 后续 S-1 帧使用 index=1 的 token
    3) 拼接 → (B, S, X, C)
    4) 展平 → (B*S, X, C)

    Returns:
        torch.Tensor: 形状 (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    # 首帧 token
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    # 其余帧 token
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
