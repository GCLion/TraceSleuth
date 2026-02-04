import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re

# 初始化语义模型
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_embedding(text):
    """安全的文本嵌入获取"""
    if not text or text.strip() == "":
        return np.zeros(384)  # 返回零向量
    
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # 多种方式尝试获取嵌入
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            embeddings = last_hidden.mean(dim=1).squeeze().numpy()
        elif hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        elif hasattr(outputs, 'logits'):
            embeddings = outputs.logits.mean(dim=1).squeeze().numpy()
        else:
            # 使用sentence-transformers作为后备
            from sentence_transformers import SentenceTransformer
            backup_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = backup_model.encode(text)
        
        return embeddings
        
    except Exception as e:
        print(f"获取嵌入失败: {e}, 文本: '{text}'")
        # 返回随机向量作为后备
        return np.random.randn(384)

def calculate_cosine_similarity(text1, text2):
    """计算余弦相似度（带错误处理）"""
    try:
        emb1 = get_sentence_embedding(text1)
        emb2 = get_sentence_embedding(text2)
        
        # 确保是2D数组
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
        
    except Exception as e:
        print(f"相似度计算错误: {e}")
        return 0.0

def get_text_from_content(content):
    """从content中提取纯文本"""
    text = ""
    for item in content:
        if item['type'] == 'text':
            text += item['text']
    return text.strip()

def extract_answer_from_text(text):
    """从文本中提取<answer>标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_sentence_embedding(text):
    """获取句子的语义嵌入向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用平均池化获取句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def calculate_cosine_similarity(text1, text2):
    """计算两个文本的余弦相似度"""
    # 获取嵌入向量
    emb1 = get_sentence_embedding(text1)
    emb2 = get_sentence_embedding(text2)
    
    # 计算余弦相似度
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def process_text_messages(text_messages, model_response):
    """处理text_messages并计算相似度"""
    try:
        # 提取assistant的文本内容
        assistant_content = None
        for message in text_messages:
            if message['role'] == 'assistant':
                assistant_content = message['content']
                break
        
        if not assistant_content:
            print("未找到assistant内容")
            return None
        
        # 提取纯文本
        assistant_text = get_text_from_content(assistant_content)
        print("Assistant文本内容:")
        print(assistant_text[:500] + "..." if len(assistant_text) > 500 else assistant_text)
        print("-" * 50)
        
        # 提取answer标签内容
        assistant_answer = extract_answer_from_text(assistant_text)
        print(f"Assistant答案: {assistant_answer}")
        print(f"Model响应答案: {model_response}")
        print("-" * 50)
        
        # 计算相似度
        if assistant_answer and model_response:
            similarity = calculate_cosine_similarity(assistant_answer, model_response)
            print(f"语义余弦相似度: {similarity:.4f}")
            return similarity
        else:
            print("无法提取有效的答案内容")
            return None
            
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 你的数据
    text_messages = ""
    
    # 假设的model_response（从item["model_response"]获取）
    model_response = "Real"  # 或者 "Fake,AI-generated", "Fake,PS-edited"
    
    # 计算相似度
    similarity = process_text_messages(text_messages, model_response)
    
    if similarity is not None:
        print(f"\n最终相似度得分: {similarity:.4f}")





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
import pandas as pd
import torchvision.transforms as T

# ==============================
# Step 1: 伪造痕迹特征提取器（仅用于训练）
# ==============================
class ForgeryDetector(nn.Module):
    def __init__(self, visual_dim=1024, forg_dim=128, num_classes=3):
        super().__init__()
        # 可选的伪造特征增强器
        self.adapter = ForgeryAdapter(dim_vis=visual_dim, dim_forg=forg_dim, hidden_dim=visual_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, visual_dim // 2),
            nn.ReLU(),
            nn.Linear(visual_dim // 2, num_classes),
        )

        # 分割头（用于 patch-level 伪造区域预测）
        self.segmenter = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.ReLU(),
            nn.Linear(visual_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, image_embeds, forgery_features):
        # 融合伪造特征
        fused = self.adapter(image_embeds, forgery_features)  # (N, D)

        # 分类
        cls_logits = self.classifier(fused.unsqueeze(0).transpose(1, 2))  # (1, N, D) => (1, D) => (1, num_classes)

        # 分割
        seg_masks = self.segmenter(fused).squeeze(-1)  # (N,)

        return cls_logits, seg_masks

# ==============================
# Step 2: 数据集准备（支持伪造mask路径）
# ==============================
class ForgeryEmbeddingDataset(Dataset):
    def __init__(self, df, vl_model, transform, device):
        self.df = df
        self.vl_model = vl_model
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        mask_path = row['mask_path']

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 获取 image_embeds
        with torch.no_grad():
            image_embeds = self.vl_model.visual(image_tensor)  # (N, D)

        # 加载伪造mask（若存在）
        if mask_path != "None":
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.transform(mask).mean(dim=0)  # (H, W)
            # resize to patch数（假设8x8）
            seg_mask = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(8, 8), mode='bilinear').view(-1)  # (64,)
        else:
            seg_mask = torch.zeros(image_embeds.shape[0])  # 全零

        # 标签
        label = torch.tensor(row['label'], dtype=torch.long)

        return image_embeds.cpu(), seg_mask.cpu(), label

# ==============================
# Step 3: 训练函数
# ==============================
def train_forgery_detector(model, dataset, epochs=5, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for image_embeds_batch, seg_masks_batch, labels_batch in dataloader:
            image_embeds_batch = image_embeds_batch.to(device)
            seg_masks_batch = seg_masks_batch.to(device)
            labels_batch = labels_batch.to(device)

            # 假设伪造特征由外部提取器提供（可以是预训练 CNN）
            forgery_features = torch.randn_like(image_embeds_batch)  # 模拟伪造特征（实际应替换）

            cls_logits, seg_preds = model(image_embeds_batch, forgery_features)

            # Loss
            cls_loss = F.cross_entropy(cls_logits, labels_batch)
            seg_loss = F.mse_loss(seg_preds, seg_masks_batch)
            loss = cls_loss + seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ==============================
# Step 4: 初始化和运行
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Qwen2.5-VL 模型（仅用于提取 image_embeds）
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("your_model_path").to(device)
for param in vl_model.parameters():
    param.requires_grad = False  # 冻结主干

# 数据集
df = pd.read_csv("mask.lst", header=None, names=["img_path", "mask_path"])
df["label"] = df["img_path"].apply(lambda x: 1 if "AutoSplice" in x else (2 if "Tp" in x else 0))

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

dataset = ForgeryEmbeddingDataset(df, vl_model, transform, device)

# 伪造检测模型
detector = ForgeryDetector(visual_dim=1024, forg_dim=128, num_classes=3).to(device)

# 开始训练
train_forgery_detector(detector, dataset, epochs=5)