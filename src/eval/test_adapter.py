from datetime import datetime
from openai import OpenAI
import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from my_vllm import AgentVLMVLLM
from tqdm import tqdm
from PIL import Image
import json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
import torch
import os  
import re  
import numpy as np
import warnings
from vllm import LLM, SamplingParams
import random
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
) 
warnings.filterwarnings("ignore")
vllm_available = True
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from my_qwen_vl import My_Qwen2_5_VL

# ======================
# Configuration
# ======================

DATA_DIR = "/train/myModel/VLM-R3-main/qwen_vl_finetune" #"/train/myModel/VLM-R3-main/vstar_bench/"
TEST_FILE = "/train/myModel/VLM-R3-main/qwen_vl_finetune/dataset/test_CASIAv2_autosplice_test.json" #"/train/myModel/VLM-R3-main/vstar_bench/test_questions.jsonl"
MODEL_PATH = "/train/myModel/VLM-R3-main/qwen_vl_finetune/0106-output_sft"
# "/train/myModel/VLM-R3-main/qwen_vl_finetune/output/rgrpo_vllm_Qwen2.5-VL-1223_0924"
# "/train/myModel/VLM-R3-main/qwen_vl_finetune/output_sft" #"/train/myModel/dropout/output_sft"
TEMP_DIR = "./crops" #"./crops_vstar_test"
VLLM_DEVICE = "cuda:0"

SEED = 42
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://serve-3ff12537-48f4-4f71-9cdf-82ecfa75cf6b-8000.uic.miya.aicloud-intl.com/v1")
VERIFY_MODEL_NAME = "qwen2.5-235b-instruct" # "qwen2.5-72b-instruct"

TEMPERATURE = 0.7#0
MIN_PIXELS = 32*28*28
MAX_PIXELS = 8192*28*28
CROP_MIN_PIXELS = 32*28*28
CROP_MAX_PIXELS = 4096*28*28

# ======================
# Initialization
# ======================
random.seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 确保有chat_template
if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    # Qwen2.5-VL标准模板
    chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|im_end|>' }}{% endif %}{% endfor %}"
    tokenizer.chat_template = chat_template
    print("已手动设置chat_template")

# llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     device_map="auto"  # 让模型自动分配到可用设备
# ).to(VLLM_DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 确保processor也有chat_template
if not hasattr(processor, 'chat_template') or processor.chat_template is None:
    processor.chat_template = tokenizer.chat_template
    
llm = My_Qwen2_5_VL.from_pretrained(
    MODEL_PATH,
    # cache_dir=training_args.cache_dir,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
).to(VLLM_DEVICE)
# processor = AutoProcessor.from_pretrained(MODEL_PATH)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
agent = AgentVLMVLLM(
    model=llm,
    processor=processor,
    temp_dir=TEMP_DIR,
    device=VLLM_DEVICE,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    temperature=TEMPERATURE,
    crop_min_pixels=CROP_MIN_PIXELS,
    crop_max_pixels=CROP_MAX_PIXELS,
)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
def evaluate_answer_similarity(student_answer, ground_truth, problem):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model=VERIFY_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a evaluation expert. Compare the student's answer with the correct answer. Output ONLY '1.0' if the student's answer matches the correct answer in meaning, or '0.0' if the student's answer does not contain a correct answer. No other output is allowed.
                    Question: {problem}\nStudent's answer: {student_answer}\nCorrect answer: {ground_truth}\nOutput only 1.0 or 0.0:"""
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        if "1.0" not in result and "0.0" not in result:
            print(f"Unexpected response from GPT: {result}")
            result = 0.0
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0


def extract_answer_tag(text):
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()
def normalize_label_to_class(ans):
    """
    将答案字符串映射到 3 类标签:
      - 'real' -> 0
      - 'fake,ai-generated' -> 1
      - 'fake,ps-edited' -> 2
    解析失败返回 None。
    """
    if ans is None:
        return None

    a = ans.strip().lower()
    a = a.strip("'\"")  # 去掉引号
    a = a.lower().replace(" ", "")
    # print(a)

    if a == "real":
        return 0
    elif a == "fake,ai-generated":
        return 1
    elif a == "fake,ps-edited":
        return 2
    else:
        # 如需更宽松，可扩展这里
        return None
def acc_verifier(full_response, ground_truth, question=None):
    """
    从 full_response 和 ground_truth 中解析 <answer>...</answer>，
    将答案映射为 0/1/2，并返回是否预测正确及相关信息。

    返回:
        is_correct: int (0 或 1)，若无法解析则为 0
        y_true: int 或 None
        y_pred: int 或 None
        has_label: bool，表示是否成功解析了真值和预测值
    """
    # 1. 提取答案内容
    gt_ans_str = extract_answer_tag(ground_truth)
    pred_ans_str = extract_answer_tag(full_response)

    # 2. 转成 0/1/2
    y_true = normalize_label_to_class(gt_ans_str)
    y_pred = normalize_label_to_class(pred_ans_str)
    # print(y_true, y_pred)

    # 若有任一无法解析，认为无效样本
    if y_true is None or y_pred is None:
        return 0, y_true, y_pred, False

    is_correct = int(y_true == y_pred)
    return is_correct, y_true, y_pred, True

from sklearn.metrics.pairwise import cosine_similarity
import re
# 初始化语义模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
def get_sentence_embedding(text):
    """获取句子的语义嵌入向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = inputs.to(VLLM_DEVICE)
    with torch.no_grad():
        outputs = llm(**inputs)
    # 使用平均池化获取句子嵌入
    # embeddings = outputs.hidden_states[-1].mean(dim=1)
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        last_hidden = outputs.hidden_states[-1]
        embeddings = last_hidden.mean(dim=1).squeeze().numpy()
    elif hasattr(outputs, 'last_hidden_state'):
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    elif hasattr(outputs, 'logits'):
        embeddings = outputs.logits.mean(dim=1).squeeze().float().cpu().numpy()
    else:
        # 使用sentence-transformers作为后备
        from sentence_transformers import SentenceTransformer
        backup_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = backup_model.encode(text)
    return embeddings
def calculate_cosine_similarity(text1, text2):
    """计算两个文本的余弦相似度"""
    # 获取嵌入向量
    emb1 = get_sentence_embedding(text1).reshape(1, -1)
    emb2 = get_sentence_embedding(text2).reshape(1, -1)
    # print()
    
    # 计算余弦相似度
    similarity = cosine_similarity(emb1, emb2)
    return similarity

import json
from collections import Counter
log_file = "./evaluation_vstar_log_{}.txt".format(timestamp)
with open(log_file, "a") as log_f:
    log_f.write(f"\n----- Evaluation Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -----\n")
# with open(TEST_FILE, "r") as f:
#     data = [json.loads(line) for line in f]
with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)   # 注意是 json.load，不是 json.loads
    
cnt = Counter()
cnt_correct = Counter()
# correct_number = 0
all_y_true = []
all_y_pred = []
valid_count = 0
from tqdm import tqdm
from cls_model import QwenFeatureDomainForgeryModel
model_path = "/train/myModel/VLM-R3-main/qwen_vl_finetune/720-Qwen2.5-VL-7B-Instruct"
model = QwenFeatureDomainForgeryModel(qwen_path=model_path, forg_dim=256, num_classes=3)
# model = QwenFeatureDomainForgeryModel(qwen_path, forg_dim=256, num_classes=3)
para_path = "/train/myModel/VLM-R3-main/qwen_vl_finetune/best_forgery_modules.pt"
state_dict = torch.load(para_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
# print("missing keys:", missing)
# print("unexpected keys:", unexpected)
model.eval()
tot_sim = 0.0
for i, item in enumerate(tqdm(data, desc="Processing items")):
    # print(item)
    image_path = DATA_DIR + item["image_path"][1:]
    ####
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": ""},  # 空文本
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    all_inputs = processor(
        text=[""],  # 空文本
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    pixel_values = all_inputs["pixel_values"]
    image_grid_thw = all_inputs["image_grid_thw"]#.to(device)
    # labels = batch["labels"].to(device)

    outputs = model(pixel_values, image_grid_thw)
    logits = outputs['logits']
    pred = torch.argmax(logits, dim=1)
    img_type = ""
    if pred == 0:
        img_type = "real"
    elif pred == 1:
        img_type = "AI-generated fake"
    elif pred == 2:
        img_type = "PS-edited fake"
    question = f"This is a {img_type} image, please analyze the details and identify which parts and areas show signs. \nYou need to first think about the reasoning process in your mind, and then provide the answer. When thinking, you should call the \"crop\" tool (format: {{\"bbox_2d\": [x1, y1, x2, y2]}}) to focus on the key areas in the image. The reasoning process and the answer are included in the <think> </think> and <answer> </answer> tags respectively. <answer> </answer> tags are only allowed to include 'fake,ai-generated', 'fake,ps-edited' or 'real'"
    ####
    
    # question = item["question"]#改为。。。
    # question = question.replace("Answer with the option's letter from the given choices directly.","").strip()
    # question = question + "\nAnswer the question with 'real', 'fake,ai-generated', or 'fake,ps-edited'."
    ground_truth = item["answer"]
    ground_truth = "<answer>" + ground_truth + "</answer>"
    
    # category = item["category"]
    # cnt[category] += 1

    _, _, full_response, img_messages, text_messages = agent.process(image_path, question)
    think = re.search(r"<think>(.*?)</think>", full_response, flags=re.IGNORECASE | re.DOTALL)
    think_ans = item["model_response"]
    try:
        think = think.group(1).strip()
        # print('think_ans:', think_ans)
        # print('think:', think)
        similarity = calculate_cosine_similarity(think_ans, think)
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        continue

    tot_sim += float(similarity)
    # print(':::', tot_sim)
    # print(text_messages)
    is_correct, y_true, y_pred, has_label = acc_verifier(full_response, ground_truth, question)
    if not has_label:
        continue  # 跳过解析失败的样本

    valid_count += 1
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)


if valid_count > 0:
    acc = accuracy_score(all_y_true, all_y_pred)

    # 多分类建议用 macro 或 weighted
    f1 = f1_score(all_y_true, all_y_pred, average="macro")
    precision = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    tot_sim /= valid_count

    print(f"Valid samples: {valid_count}")
    print(f"ACC: {acc:.4f}")
    print(f"F1:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Similarity: {tot_sim:.4f}")
else:
    print("No valid samples to evaluate.")
    acc = f1 = precision = recall = 0.0

# 写 log
with open(log_file, "a", encoding="utf-8") as log_f:
    log_f.write("model_path: {}\n".format(MODEL_PATH))
    log_f.write("Valid samples: {}\n".format(valid_count))
    log_f.write("ACC: {:.4f}\n".format(acc))
    log_f.write("F1: {:.4f}\n".format(f1))
    log_f.write("Precision: {:.4f}\n".format(precision))
    log_f.write("Recall: {:.4f}\n".format(recall))
    log_f.write("-" * 50 + "\n")



# with open(log_file, "a") as log_f:
#     log_f.write("model_path: {}\n".format(MODEL_PATH))
#     log_f.write("Valid samples: {}\n".format(valid_count))
#     log_f.write("ACC: {:.4f}\n".format(acc))
#     log_f.write("F1: {:.4f}\n".format(f1))
#     log_f.write("Precision: {:.4f}\n".format(precision))
#     log_f.write("Recall: {:.4f}\n".format(recall))
#     log_f.write("TP: {}, FP: {}, TN: {}, FN: {}\n".format(tp, fp, tn, fn))
#     log_f.write("-" * 50 + "\n")

# with open(log_file, "a") as log_f:
#     log_f.write("model_path: {}\n".format(MODEL_PATH))
#     print(f"## DEBUG: correct ratio: {correct_number/len(data)} = {correct_number}/{len(data)}")
#     log_f.write(f"\nFinal Correct Ratio: {correct_number/len(data)} = {correct_number}/{len(data)}\n")
#     print("## DEBUG: cnt:", cnt)
#     print("## DEBUG: cnt_correct:", cnt_correct)
#     log_f.write(f"\nCategory Counts:\n{cnt}\n")
#     log_f.write(f"\nCategory Correct Counts:\n{cnt_correct}\n")

#     r1 = acc_verifier(full_response, ground_truth, question)
#     if r1 == 1.0:
#         correct_number += 1
#         cnt_correct[category] += 1
#     print(f"## DEBUG: correct ratio: {correct_number/(i+1)} = {correct_number}/{i+1}")
    
#     if r1 != 1.0 or random.randint(0,1)==0:
#         # Log full_response and reward details to file
#         with open(log_file, "a") as log_f:
#             log_f.write(f"\nIteration {i}:\n")
#             log_f.write(f"Image Path: {image_path}\n")
#             log_f.write(f"Question: {question}\n")
#             log_f.write(f"Category: {category}\n")
#             log_f.write("Full Response:\n")
#             log_f.write(full_response + "\n")
#             log_f.write(f"Reward: {r1}\n")
#             log_f.write("Ground Truth:\n")
#             log_f.write(ground_truth + "\n")
#             log_f.write("-" * 60 + "\n")