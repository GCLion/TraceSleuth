import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor, AutoModel  # 名称需根据实际 Qwen2.5-VL 实现调整
import random
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureForgeryEncoder(nn.Module):
    """
    输入: feat_vis [N_patches, D_vis]
    输出: feat_forg [N_patches, D_forg]
    在 ViT 特征空间里提取“伪造风格特征”，保证 patch 数绝对一致
    """
    def __init__(self, dim_vis, dim_forg=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_vis, dim_vis),
            nn.ReLU(inplace=True),
            nn.Linear(dim_vis, dim_forg),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_vis):
        return self.mlp(feat_vis)  # [N_patches, dim_forg]

class ForgeryAdapter(nn.Module):
    def __init__(self, dim_vis, dim_forg, hidden_dim=None):
        super().__init__()
        self.dim_vis = dim_vis
        self.dim_forg = dim_forg
        self.hidden_dim = hidden_dim or dim_vis

        self.proj_forg = nn.Linear(dim_forg, dim_vis)
        self.weight_gate = nn.Parameter(torch.zeros(dim_vis))
        if self.hidden_dim != dim_vis:
            self.fuse_proj = nn.Linear(dim_vis, self.hidden_dim)
        else:
            self.fuse_proj = nn.Identity()
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, feat_vis, feat_forg):
        """
        feat_vis: [N, dim_vis]
        feat_forg: [N, dim_forg]
        """
        assert feat_vis.shape[0] == feat_forg.shape[0], \
            f"patch数不一致: vis={feat_vis.shape[0]}, forg={feat_forg.shape[0]}"

        forg_proj = self.proj_forg(feat_forg)  # [N, dim_vis]
        alpha = torch.sigmoid(self.weight_gate)  # [dim_vis]
        fused = feat_vis + alpha * forg_proj     # [N, dim_vis]
        out = self.fuse_proj(fused)
        out = self.norm(out)
        return out  # [N, hidden_dim]

class ForgeryClassifierHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, fused_feats, batch_sizes):
        """
        fused_feats: [N_total_patches, hidden_dim]
        batch_sizes: list[int] 每张图的 patch 数
        """
        logits_list = []
        start = 0
        for n_patches in batch_sizes:
            end = start + n_patches
            img_feats = fused_feats[start:end]          # [n_patches_i, hidden_dim]
            global_feat = img_feats.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            logits = self.classifier(global_feat)       # [1, num_classes]
            logits_list.append(logits)
            start = end
        return torch.cat(logits_list, dim=0)            # [B, num_classes]

class QwenFeatureDomainForgeryModel(nn.Module):
    def __init__(self, qwen_path, forg_dim=256, num_classes=3):
        super().__init__()
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path)
        self.visual = base_model.visual

        # # 冻结 ViT
        # for p in self.visual.parameters():
        #     p.requires_grad = False
        for n, p in self.visual.named_parameters():
            p.requires_grad = False

        # # 只解冻 merger
        # for n, p in self.visual.merger.named_parameters():
        #     p.requires_grad = True

        dim_vis = self.visual.config.out_hidden_size  # e.g. 1176

        self.forgery_encoder = FeatureForgeryEncoder(dim_vis=dim_vis, dim_forg=forg_dim)
        self.adapter = ForgeryAdapter(dim_vis=dim_vis, dim_forg=forg_dim, hidden_dim=dim_vis)
        self.classifier = ForgeryClassifierHead(hidden_dim=dim_vis, num_classes=num_classes)

    def forward(self, pixel_values, image_grid_thw, labels=None):
        """
        pixel_values: [B, 3, H, W]（processor 处理后的图）
        image_grid_thw: [B, 3]，每张图 (t, h_grid, w_grid)
        labels: [B]，0/1/2
        """
        # print(type(pixel_values)) # <class 'torch.Tensor'>
        # print(pixel_values.shape) # torch.Size([504, 1176])
        # print(pixel_values)       # 
        pixel_values = pixel_values.unsqueeze(0)
        
        B = pixel_values.size(0)
        # print(B)
        # B = 1

        # 1) ViT 语义特征 A
        with torch.no_grad():
            feat_vis = self.visual(pixel_values, grid_thw=image_grid_thw)  # [N_total_patches, dim_vis]

        # 2) 伪造特征 B：直接在 feat_vis 上算，天然 patch 对齐
        feat_forg = self.forgery_encoder(feat_vis)                         # [N_total_patches, forg_dim]

        # 3) Adapter 融合 → C
        fused_feats = self.adapter(feat_vis, feat_forg)                    # [N_total_patches, dim_vis]
        
        # fused_feats = feat_vis                     # [N_total_patches, dim_vis]

        # 4) 按图像聚合做 3 分类
        batch_sizes = []
        # print(type(image_grid_thw)) #<class 'torch.Tensor'>
        # print(image_grid_thw.shape) #torch.Size([1, 3])
        # print(image_grid_thw)       #tensor([[ 1, 18, 26]])

        # if image_grid_thw.dim() == 1:
        #     # 如果只有1个元素，直接使用它
        #     if len(image_grid_thw) == 1:
        #         t, h, w = image_grid_thw[0].tolist()  # 使用索引0而不是i
        #     else:
        #         # 如果有多个元素，需要使用正确的索引
        #         # 这里需要根据你的逻辑确定使用哪个索引
        #         t, h, w = image_grid_thw[0].tolist()  # 或者根据实际情况调整
        #     batch_sizes.append(t * h * w)
        # else:
        #### ori
        for i in range(B):
            t, h, w = image_grid_thw[i].tolist()
            batch_sizes.append(t * h * w)  # 这一张图的 patch 数

        logits = self.classifier(fused_feats, batch_sizes)                 # [B, num_classes]

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss

        return out

# 在 cls_model.py 的第133行附近修改
# def forward(self, pixel_values, image_grid_thw):
#     try:
#         # 检查 image_grid_thw 的维度
#         if image_grid_thw.dim() == 1:
#             # 如果只有1个元素，直接使用它
#             if len(image_grid_thw) == 1:
#                 t, h, w = image_grid_thw[0].tolist()  # 使用索引0而不是i
#             else:
#                 # 如果有多个元素，需要使用正确的索引
#                 # 这里需要根据你的逻辑确定使用哪个索引
#                 t, h, w = image_grid_thw[0].tolist()  # 或者根据实际情况调整
#         else:
#             # 原来的逻辑
#             t, h, w = image_grid_thw[i].tolist()
#     except IndexError as e:
#         print(f"IndexError: image_grid_thw shape: {image_grid_thw.shape}, i: {i}")
#         raise e


