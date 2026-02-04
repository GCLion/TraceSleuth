import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

# class MyModuleA(nn.Module):
#     """示例：处理视觉特征 image_embeds 的模块"""
#     def __init__(self, dim_vis):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_vis, dim_vis),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, image_embeds):
#         # image_embeds: [N_image_tokens, dim_vis]
#         return self.mlp(image_embeds)

# class MyModuleB(nn.Module):
#     """示例：处理 LLM hidden_states 的模块"""
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#         )

#     def forward(self, hidden_states):
#         # hidden_states: [B, L, hidden_size]
#         return self.mlp(hidden_states)



# qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_args.model_name_or_path,
#     attn_implementation=attn_implementation,
#     torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
# )

# # 依据 config 获取维度
# dim_vis = qwen.visual.config.out_hidden_size
# hidden_size = qwen.config.hidden_size

# # 创建并加载你预训练好的权重（这里只是示例）
# moduleA = MyModuleA(dim_vis)
# moduleB = MyModuleB(hidden_size)

# # 比如加载你自己的 checkpoint
# # moduleA.load_state_dict(torch.load('moduleA.pt'))
# # moduleB.load_state_dict(torch.load('moduleB.pt'))

# # 是否冻结这些模块自己控制
# for p in moduleA.parameters():
#     p.requires_grad = False  # 例如完全冻结
# for p in moduleB.parameters():
#     p.requires_grad = False  # 或者可以设置为 True 做联合微调

# # 挂到 Qwen 模型上，方便在 forward 里调用
# qwen.moduleA = moduleA
# qwen.moduleB = moduleB



# class QwenWithCustomModules(Qwen2_5_VLForConditionalGeneration):
#     def __init__(self, config, moduleA=None, moduleB=None):
#         super().__init__(config)
#         self.moduleA = moduleA
#         self.moduleB = moduleB

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         pixel_values=None,
#         pixel_values_videos=None,
#         image_grid_thw=None,
#         video_grid_thw=None,
#         rope_deltas=None,
#         cache_position=None,
#         second_per_grid_ts=None,
#     ):
#         # ====== 原始前半部分 ======
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if inputs_embeds is None:
#             inputs_embeds = self.model.embed_tokens(input_ids)
#             if pixel_values is not None:
#                 pixel_values = pixel_values.type(self.visual.dtype)
#                 image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # [N_image_tokens, dim_vis]

#                 # ====== 在这里插入 ModuleA 处理视觉特征 ======
#                 if hasattr(self, "moduleA") and self.moduleA is not None:
#                     with torch.no_grad():  # 如果你不想训练 moduleA
#                         image_embeds = self.moduleA(image_embeds)

#                 n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#                 n_image_features = image_embeds.shape[0]
#                 if n_image_tokens != n_image_features:
#                     raise ValueError(
#                         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                     )

#                 mask = input_ids == self.config.image_token_id
#                 mask_unsqueezed = mask.unsqueeze(-1)
#                 mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#                 image_mask = mask_expanded.to(inputs_embeds.device)

#                 image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#                 inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#             # 视频部分略（同理可插 ModuleA_Video）
#             if pixel_values_videos is not None:
#                 ...
#         # ====== RoPE / position_ids 原逻辑略 ======
#         ...
#         outputs = self.model(
#             input_ids=None,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs[0]  # [B, L, hidden_size]

#         # ====== 在这里插入 ModuleB 处理 LLM hidden_states ======
#         if hasattr(self, "moduleB") and self.moduleB is not None:
#             with torch.no_grad():  # 如果 moduleB 不参与训练
#                 hidden_states = self.moduleB(hidden_states)  # [B, L, hidden_size]

#         logits = self.lm_head(hidden_states)

#         # ====== 后面 loss / return_dict 保持不变 ======
#         loss = None
#         if labels is not None:
#             logits = logits.float()
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return Qwen2_5_VLCausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             rope_deltas=self.rope_deltas,
#         )



# 1. 加载原 config
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_args.model_name_or_path)

# 2. 创建自定义模型，并挂模块
model = QwenWithCustomModules(
    config=config,
    moduleA=moduleA,
    moduleB=moduleB,
)

# 3. 再从预训练权重加载（只覆盖视觉/LLM部分）
base_state = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16 if training_args.bf16 else None,
).state_dict()

missing, unexpected = model.load_state_dict(base_state, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))