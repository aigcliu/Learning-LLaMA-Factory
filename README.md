## 7 天速通 + 动手实战路线图（总览）

> **注意：本文档内容均由 [Cursor](https://cursor.sh/) AI 代码编辑器生成，用于 LLaMA-Factory 学习指导。**

### 目录

### 目录
- [总览](#7-天速通--动手实战路线图总览)
- [学习目标与最终产出（7 天）](#学习目标与最终产出-7-天)
- [全周总追踪清单（高层）](#全周总追踪清单高层)
- [Day 1](#day-1架构鸟瞰与跑通24-小时)
- [Day 2](#day-2数据与对话模板核心46-小时)
- [Day 3](#day-3模型加载peft量化补丁4-小时)
- [Day 4](#day-4训练框架全景sftdpoktopprmpt6-小时)
- [Day 5](#day-5推理与服务hfvllmsglang--webui4-小时)
- [Day 6](#day-6评估与多机多卡容器4-小时)
- [Day 7](#day-7扩展实战新增模板模型数据处理器6-小时)
- [每日追踪模板](#每日追踪模板复制填充)
- [常见坑与规避](#常见坑与规避)
- [速查命令](#速查命令常用)
- [建议笔记结构](#建议笔记结构)

- Day 1：架构鸟瞰与跑通 — 阅读 README/入口脚本；跑通 LoRA SFT 10 步、CLI 对话与导出。
- Day 2：数据与模板 — 精读 `template.py/formatter.py`；对拍 Qwen/Llama3/DeepSeek3 模板；新增 `my_llama3` 与 `my_demo` 并各训 10 步。
- Day 3：模型加载/PEFT/量化/补丁 — 读 `loader/adapter/patcher` 与 `model_utils/*`；在 `constants.py` 注册新模型；做一次 QLoRA 显存/吞吐对比。
- Day 4：训练框架全景 — 跑 SFT/DPO/RM 各 50 步；理解 `hparams/*` 与 `callbacks`；记录收敛曲线。
- Day 5：推理与服务 — 切换 HF/vLLM/SGLang；使用 WebUI 加载权重多轮对话；对比吞吐/延迟。
- Day 6：评估与多机多卡/容器 — 跑 MMLU/CEval/CMMLU 子集；尝试 Accelerate/Deepspeed；API 部署与容器化要点。
- Day 7：扩展实战 — 新增模板/模型/数据处理器（三选一或多选）；端到端复现并形成最小 MR 草稿。

配套产出：每日笔记、最小脚本、对比报告、端到端一键脚本与可提交改动草稿。

（详细步骤与命令见下文各 Day 小节）

---

## 学习目标与最终产出（7 天）

- **目标**：在 7 天内系统掌握 LLaMA-Factory 的架构与代码细节，具备二次开发能力（新增模型/模板/数据处理/训练法）。
- **最终产出**：
  - 每日学习与代码阅读笔记（建议建一个 notes/ 目录）
  - 3 条可复现实验流水线（SFT/DPO/RM 任三种）
  - 1 个新增能力（新模板/新模型映射/新数据处理器），含示例与文档
  - 端到端脚本：训练 → 推理 → 导出/合并 → 评估（可一键复现）

---

## 全周总追踪清单（高层）

- [ ] 读完并标注关键处：`README_zh.md`、`examples/README_zh.md`
- [ ] 跑通一次 LoRA SFT（10~50 步快速验证）
- [ ] 跑通一次推理/导出/合并
- [ ] 做 3 次短训（SFT/DPO/RM 各一次）
- [ ] HF 与 vLLM 推理对比（吞吐/延迟/显存）
- [ ] 新增一个模板或模型映射，并可正常 chat
- [ ] 接入一个自定义数据集并训练 10 步
- [ ] 至少 1 个评估集（MMLU/CEval/CMMLU）跑通并出表

---

## Day 1：架构鸟瞰与跑通（2~4 小时）

- **逐小时计划**：
  - 第 1 小时：快速过一遍 `README_zh.md` 与 `examples/README_zh.md`，标注“安装/快速开始/模型与模板表/多 GPU”。
  - 第 2 小时：读 `src/llamafactory/cli.py` 的 `main()` 调用链；定位子命令 `train/chat/export/webui/api` 的分发位置；对照命令参数与 YAML 覆盖逻辑。
  - 第 3 小时：扫 `src/train.py`、`src/webui.py`、`src/api.py` 的入口，理解如何调用内部模块。
  - 第 4 小时：执行一轮最小 SFT/Chat/Export（见下），记录日志与产物路径。

- **阅读**：
  - `README_zh.md`（如何使用/模型/依赖/快速开始）
  - `examples/README_zh.md`（多 GPU、导出/合并）
  - 入口：`src/llamafactory/cli.py`、`src/train.py`、`src/webui.py`、`src/api.py`
  - 结构速览：`src/llamafactory/` 下 `data/`、`model/`、`train/`、`chat/`、`webui/`、`extras/`、`hparams/`、`eval/`

- **当日 P0/P1/P2 覆盖点**：
  - P0：`src/llamafactory/cli.py`、`src/llamafactory/hparams/*`（参数如何注入与合并）
  - P1：快速过 `examples/*` 的 YAML 范式
  - P2：略看 `README_zh.md` 依赖/硬件表，标注后续需验证项

- **当日代码精读锚点**：
  - `src/llamafactory/cli.py: main()` 与子命令装配
  - `src/llamafactory/hparams/{model_args,data_args,training_args,generating_args,finetuning_args}.py`：参数定义与校验

- **当日推荐阅读顺序（微观）**：`README_zh.md` → `examples/README_zh.md` → `cli.py: main()` → `hparams/*`
- **实操命令（前台）**：
```bash
cd <PROJECT_ROOT>
export USE_MODELSCOPE_HUB=1

# 快速 SFT（10 步）
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
  max_steps=10 logging_steps=1 output_dir=saves/day1/llama3_lora

# 推理（命令行交互，/exit 退出）
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml

# 合并/导出（LoRA 合并示例）
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml \
  output_dir=saves/day1/export
```
- **验收（期望输出）**：
  - 训练日志每步打印 loss；`saves/day1/llama3_lora` 生成 LoRA 权重
  - chat 可多轮对话，`/exit` 正常退出
  - `saves/day1/export` 下出现导出模型文件夹
- **产出**：入口与调用链思维导图/笔记；脚本 `scripts/day1_quickstart.sh`

---

## Day 2：数据与对话模板（核心，4~6 小时）

- **逐小时计划**：
  - 第 1 小时：精读 `src/llamafactory/data/template.py` 顶部的 `class Template` 字段与 `encode_oneturn/encode_multiturn`；理解 `stop_words/thought_words/efficient_eos` 的作用。
  - 第 2 小时：精读 `parse_template()` 与 `get_template_and_fix_tokenizer()`，理解如何从 `tokenizer.chat_template` 自动抽取模板与修正 tokenizer；用 3 个不同模型做对拍。
  - 第 3 小时：阅读 `formatter.py` 中 `StringFormatter/FunctionFormatter/ToolFormatter/EmptyFormatter`，理解 `slots` 与工具调用的插槽机制。
  - 第 4 小时：扫 `data/processor/supervised.py` 与 `collator.py`，理解 messages 生成与打包；看 `converter.py` 的样例转换。
  - 第 5 小时：新增 `my_llama3` 模板并做 10 步训练验证；记录渲染结果与差异。
  - 第 6 小时：新增 `my_demo` 数据集（本地 JSONL + `dataset_info.json` 注册），做 10 步训练验证。

- **阅读**：
  - 模板：`src/llamafactory/data/template.py`（`register_template`、`Template.encode_*`、`parse_template`）
  - 格式化器：`src/llamafactory/data/formatter.py`（`StringFormatter/FunctionFormatter/ToolFormatter`）
  - 数据：`src/llamafactory/data/processor/*.py`、`collator.py`、`converter.py`、`data_utils.py`
  - 数据注册：`data/dataset_info.json`

- **当日 P0/P1/P2 覆盖点**：
  - P0：`data/template.py`、`data/formatter.py`、`data/processor/supervised.py`、`data/collator.py`
  - P1：`data/converter.py` 与多格式兼容、工具调用 `ToolFormatter`
  - P2：`tests/data/test_template.py` 了解期望行为

- **当日代码精读锚点**：
  - `Template.encode_oneturn/encode_multiturn`、`parse_template()`、`get_template_and_fix_tokenizer()`
  - `StringFormatter/FunctionFormatter/ToolFormatter/EmptyFormatter` 的 slots 组合
  - `processor/supervised.py` 的样本到 messages 过程、`collator.py` 的 batch 拼接

- **当日推荐阅读顺序（微观）**：`template.py` → `formatter.py` → `processor/supervised.py` → `collator.py` → `converter.py`
- **模板渲染对拍（Qwen/Llama3/DeepSeek3）**：
```python
from transformers import AutoTokenizer

def show(mid):
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    msgs = [{"role":"user","content":"你好"}]
    print(mid, "\n", tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "\n")

show("qwen/Qwen1.5-0.5B-Chat")
show("meta-llama/Meta-Llama-3-8B-Instruct")
show("deepseek-ai/DeepSeek-V3-Chat")
```
- **新增“自定义模板”并验证**：
  - 在 `template.py` 复制 `llama3` 的注册，改名 `my_llama3`（保留 stop_words/format_prefix 等）
```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
  template=my_llama3 max_steps=10 output_dir=saves/day2/my_llama3
```
- **新增“自定义数据集”并验证**：
  - 准备 `data/my_demo.jsonl`（标准 messages 或问答格式）并在 `data/dataset_info.json` 注册条目 `my_demo`
```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
  dataset=my_demo max_steps=10 output_dir=saves/day2/my_data
```
- **验收**：三模型模板渲染差异明确；`my_llama3` 和 `my_demo` 均可训练 10 步
- **产出**：模板对拍记录与图示；`my_llama3` 最小 MR 草稿

**增强：模板高级用法（结合项目能力）**
- 工具调用对拍（GLM4/ChatML/Qwen/Llama3）：
  - 参考文件：`src/llamafactory/data/formatter.py`（`ToolFormatter`）、`src/llamafactory/data/template.py` 中 `chatglm3/qwen/llama3` 模板注册
  - 验证建议：准备一条包含 `function_call/tool_response` 的 messages，在不同模板下渲染并比较工具槽位
- 思维标签与 ReasoningTemplate：
  - 参考：`ReasoningTemplate` 与 `<think>...</think>` 插入、`efficient_eos/replace_eos` 行为
  - 验证建议：渲染包含思维标签的 assistant 片段，检查停止词与替换策略
- 多模态模板最小验证：
  - 参考：`qwen2_vl/llava_next/*` 模板与 `mm_plugin`
  - 命令（仅渲染验证，不必下载权重）：在 Python 中传入 `messages=[{"role":"user","content":"<image>请描述图像"}]`，渲染并检查 `<image>` 或相应占位符是否插入

---

## Day 3：模型加载/PEFT/量化/补丁（4 小时）

- **逐小时计划**：
  - 第 1 小时：阅读 `src/llamafactory/model/loader.py` 中模型与 tokenizer 的加载流程（`trust_remote_code`、`device_map`、`quantization_config`、`auto_resume`）。
  - 第 2 小时：阅读 `adapter.py`（LoRA/QLoRA 注入）、`patcher.py`（对特定模型族注入补丁），扫 `model_utils/*`（rope/kv_cache/moe/quantization/packing）。
  - 第 3 小时：在 `extras/constants.py` 注册一个新模型映射，指定 `template`，并用 chat 验证可用。
  - 第 4 小时：跑一次 QLoRA（AWQ/BnB 任一），记录显存、吞吐与速度对比。

- **阅读**：
  - `src/llamafactory/model/loader.py`、`adapter.py`、`patcher.py`、`model_utils/*`
  - `src/llamafactory/extras/constants.py`（模型分组与默认 `template`）
- **实操**：在 `constants.py` 注册一个小模型映射（HF/ModelScope），指定正确模板并验证 chat；跑一次 QLoRA 对比显存/速度
```bash
# QLoRA 示例（可选任一）
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml \
  max_steps=20 output_dir=saves/day3/qlora
```
- **验收**：新模型映射可 chat；QLoRA 显存较 LoRA 明显下降（记录显存/吞吐）
- **产出**：加载/量化/PEFT 组合流程图；新模型映射 MR 草稿

**增强：量化矩阵与兼容性（结合项目示例）**
- 训练侧量化对比（任选其三）：
```bash
# AWQ（示例）
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml \
  max_steps=50 output_dir=saves/day3/awq

# GPTQ（示例）
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml \
  max_steps=50 output_dir=saves/day3/gptq

# AQLM（示例）
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml \
  max_steps=50 output_dir=saves/day3/aqlm
```
- 推理端一致性检查：
  - HF 后端：加载三种量化产物做相同 prompt 推理
  - vLLM 后端：优先测试 README 推荐支持的量化；不兼容则记录限制与回落方案
- 输出：表格含（量化类型、显存、吞吐、是否可在 vLLM 运行、注意事项）

- **当日 P0/P1/P2 覆盖点**：
  - P0：`model/loader.py`、`model/adapter.py`、`model_utils/quantization.py|rope.py|kv_cache.py`
  - P1：`model/patcher.py` 及特定模型族补丁逻辑
  - P2：`extras/constants.py` 的模型分组与默认模板（便于产品化）

- **当日代码精读锚点**：
  - `loader.py` 的权重/分词器加载、`device_map`、量化配置
  - `adapter.py` 的 LoRA/QLoRA 注入点与与 dtype 配置
  - `model_utils/quantization.py` 关键路径与与 PEFT 的组合

- **当日推荐阅读顺序（微观）**：`loader.py` → `adapter.py` → `model_utils/*` → `patcher.py` → `extras/constants.py`

---

## Day 4：训练框架全景（SFT/DPO/KTO/PPO/RM/PT，6 小时）

- **逐小时计划**：
  - 第 1 小时：通览 `src/llamafactory/hparams/*.py`（`model_args/data_args/training_args/generating_args/finetuning_args`），理解参数来源与落地。
  - 第 2 小时：阅读 `train/sft/trainer.py` 与 `tuner.py`、`trainer_utils.py` 的核心钩子与 `compute_loss`；跑 10 步 dry-run。
  - 第 3 小时：阅读 `train/dpo/trainer.py` 与 `train/rm/trainer.py` 的损失定义与数据形状假设；各跑 10 步 dry-run。
  - 第 4 小时：阅读 `train/ppo/trainer.py` 和 `train/kto/trainer.py` 的核心差异；理解 `callbacks.py` 的保存/日志回调。
  - 第 5 小时：三类短训各 50 步；打开/关闭 `flash_attn/rope_scaling/neftune`，记录差异。
  - 第 6 小时：整理对比表与问题清单，形成“参数建议”。

- **阅读**：
  - `src/llamafactory/train/*/trainer.py`、`tuner.py`、`trainer_utils.py`、`callbacks.py`
  - `src/llamafactory/hparams/*.py`（参数落地与校验）
- **实操**：三类短训各 50 步，并记录日志
```bash
# SFT
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
  max_steps=50 output_dir=saves/day4/sft

# DPO
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml \
  max_steps=50 output_dir=saves/day4/dpo

# 奖励模型（RM）
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml \
  max_steps=50 output_dir=saves/day4/rm
```
- **验收**：三方法 loss/metric 收敛合理；能解释 `flash_attn`、`rope_scaling`、`neftune` 影响
- **产出**：显存/速度/收敛对比表；常见错误排查清单

- **当日 P0/P1/P2 覆盖点**：
  - P0：`train/sft|dpo|rm/trainer.py`、`tuner.py`、`trainer_utils.py`、`callbacks.py`、`hparams/*`
  - P1：`train/ppo|kto/trainer.py` 的实现差异与适用场景
  - P2：`tests/train/test_sft_trainer.py`（了解期望行为）

- **当日代码精读锚点**：
  - 各 `trainer.py` 的 `compute_loss/training_step/evaluate` 关键路径
  - `tuner.py` 的 Trainer 组装与策略选择
  - `callbacks.py` 的保存、日志与自定义回调点

- **当日推荐阅读顺序（微观）**：`hparams/*` → `train/sft/trainer.py` → `tuner.py` → 其他 trainer → `callbacks.py`

**增强：性能深潜（结合项目开关）**
- FlashAttention-2：在 YAML 中设置 `flash_attn: fa2`
- Liger Kernel：`enable_liger_kernel: true`（需安装对应 extras）
- Unsloth（长序列/提速）：`use_unsloth: true`
- Packing/无污染打包：`neat_packing: true`
- FSDP+QLoRA：使用 `examples/accelerate/fsdp_config.yaml` 或 `examples/accelerate/fsdp_config_offload.yaml`
- 产出：固定模型/批大小/长度，切换上述开关分别记录吞吐/显存/稳定性，形成推荐组合

---

## Day 5：推理与服务（HF/vLLM/SGLang + WebUI，4 小时）

- **逐小时计划**：
  - 第 1 小时：阅读 `chat/base_engine.py`（请求到生成的通道）、`hf_engine.py` 的 `chat()` 生成参数映射。
  - 第 2 小时：阅读 `vllm_engine.py` 与 `sglang_engine.py` 的差异（批量/并发/stop 的处理）；准备同一权重的后端对比清单。
  - 第 3 小时：跑 HF 与 vLLM 推理对比（固定 prompt 与 num_calls），记录吞吐/延迟/显存。
  - 第 4 小时：启动 WebUI，检查参数透传与多轮对话；若多模态，测试图片/视频输入。

- **阅读**：
  - `src/llamafactory/chat/`（`hf_engine.py`、`vllm_engine.py`、`sglang_engine.py`）
  - `src/llamafactory/api/`（`app.py`、`protocol.py`）与 `webui/`
- **实操**：同一权重切换后端并对比
```bash
# HF 推理
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml infer_backend=huggingface

# vLLM 推理
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml \
  infer_backend=vllm vllm_enforce_eager=true

# WebUI（浏览器界面）
llamafactory-cli webui
```
- **验收**：HF vs vLLM 吞吐/延迟对比表；WebUI 可加载自训权重并多轮对话
- **产出**：推理对比报告与推荐配置

- **当日 P0/P1/P2 覆盖点**：
  - P0：`chat/base_engine.py`、`chat/hf_engine.py`、`chat/vllm_engine.py`
  - P1：`chat/sglang_engine.py` 与 `api/*`、`webui/*`
  - P2：脚本 `scripts/api_example/*` 用于验证协议

- **当日代码精读锚点**：
  - `BaseEngine.chat()` 的调度流程与生成参数映射
  - HF vs vLLM 的不同 stop/并发与流式处理
  - `api/app.py` 的路由与 `protocol.py` 的数据结构

- **当日推荐阅读顺序（微观）**：`chat/base_engine.py` → `chat/hf_engine.py` → `chat/vllm_engine.py` → `api/*` → `webui/*`

**增强：vLLM 实战调优（结合项目命令）**
- API 部署（参考 README）：
```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml \
  infer_backend=vllm vllm_enforce_eager=true
```
- 参数建议：`gpu_memory_utilization`、`max_num_seqs`、`block_size`、`max_model_len`
- 压测建议：固定 prompt 与并发（简单并行脚本发起多请求），记录 QPS、p50/p95 延迟

---

## Day 6：评估与多机多卡/容器（4 小时）

- **逐小时计划**：
  - 第 1 小时：阅读 `eval/evaluator.py` 与 `eval/template.py`，明确评估样式/模板拼接。
  - 第 2 小时：跑一个评估子集（MMLU/CEval/CMMLU 任一），导出表格；核对 stop/eos 设置。
  - 第 3 小时：阅读 `examples/accelerate/*`、`examples/deepspeed/*` 配置；尝试 ZeRO-2 或 FSDP；记录吞吐/显存。
  - 第 4 小时：浏览 `docker/*` 与 `README_zh.md` 的依赖表，记下 CUDA/NPU/ROCm 的差异点与镜像命令。

- **阅读**：
  - 评估：`src/llamafactory/eval/evaluator.py`、`eval/template.py`
  - 多机多卡：`examples/accelerate/*`、`examples/deepspeed/*`
  - Docker：`docker/*` 与 `README_zh.md` 依赖表
- **实操**：
```bash
# vLLM 部署 OpenAI API（示例）
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml \
  infer_backend=vllm vllm_enforce_eager=true

# Deepspeed ZeRO-2（示例）
llamafactory-cli train examples/deepspeed/ds_z2_offload_config.json \
  examples/train_lora/llama3_lora_sft.yaml max_steps=50
```
- **验收**：至少 1 个评估集跑通并出表；多卡训练成功并记录吞吐/显存差异
- **产出**：评估脚本与表格；多卡启动与对比文档

- **当日 P0/P1/P2 覆盖点**：
  - P0：`eval/evaluator.py`、`examples/accelerate/*|deepspeed/*`
  - P1：`eval/template.py`（评估输入拼接）
  - P2：`docker/*`、依赖/硬件表（部署）

- **当日代码精读锚点**：
  - `evaluator.py` 的评估数据流与指标计算路径
  - Accelerate/Deepspeed 配置项与常见显存/吞吐影响参数

- **当日推荐阅读顺序（微观）**：`eval/evaluator.py` → `eval/template.py` → `examples/accelerate/*` → `examples/deepspeed/*` → `docker/*`

**增强：tests 与评测稳定性**
- 运行部分单测（需安装 pytest）：
```bash
pytest -q tests/data/test_template.py -k basic --maxfail=1
pytest -q tests/e2e/test_train.py -k sft --maxfail=1
```
- 评测稳定性：固定 seeds、温度、停止符与模板，多次运行取均值/方差
- 产出：评测汇总表（均值±方差），评测脚本与固定参数清单

---

## Day 7：扩展实战（新增模板/模型/数据处理器，6 小时）

- **逐小时计划**：
  - 第 1 小时：确定扩展方向与验收用例（模板/模型/数据处理器），列测试脚本与评估标准。
  - 第 2 小时：完成代码改动（如新增 `register_template(...)`、`constants.py` 模型组或新 `processor`）。
  - 第 3 小时：小样本训练 100 步；修正模板与 tokenizer 兼容性问题。
  - 第 4 小时：推理/导出/评估全链路验证；补充 README 片段与 YAML 示例。
  - 第 5 小时：写最小 MR 草稿（改动点/设计说明/测试记录/复现实验脚本）。
  - 第 6 小时：复盘与完善：优化默认参数、边界用例、错误提示。

- **目标**：完成一次“从 0 到 1”的可合入改动（任选其一或多项）
  - 新模板：在 `template.py` 新增 `register_template(name="your_template", ...)`，配好 `stop_words/format_prefix`；用 Day2 的自定义数据训 100 步并 chat
  - 新模型：在 `extras/constants.py` 注册新模型组与默认模板；`chat` 成功
  - 新数据处理器：在 `data/processor/` 新增 Processor，接非标 JSON 并可训练
- **验收**：端到端脚本一键复现（训练→推理→导出→评估）；形成最小 MR 草稿（代码+文档+示例 YAML）
- **产出**：你的“专家级”扩展 PR 草稿

- **当日 P0/P1/P2 覆盖点**：
  - P0：紧扣扩展方向的核心代码（模板/模型组/processor 任一）
  - P1：关联链路（训练/推理/评估/导出）
  - P2：文档与样例 YAML 的可复现性

- **当日代码精读锚点**：
  - 模板分支：`template.py` 注册、`parse_template()` 兼容性
  - 模型分支：`extras/constants.py`、`model/loader.py` 加载细节
  - 数据分支：`data/processor/*`、`collator.py` 的输入/输出契约

- **当日推荐阅读顺序（微观）**：先改核心分支 → 跑最小训练 → 通路验证（chat/export/eval） → 文档/示例补齐

**增强：PR 质量门槛（结合仓库规范）**
- 改动内容：代码 + 文档（README 片段）+ 示例 YAML + 最小测试（模板渲染/encode 或训练 1-5 步 dry-run）
- 兼容性说明：默认模板/数据格式/推理后端的影响点
- 复现材料：一键脚本 + 期望输出（含日志/产物目录树）

---

## 每日追踪模板（复制填充）

### 日期：YYYY-MM-DD
- **阅读**：文件/范围/要点记录
- **实验**：命令/参数/耗时/显存/吞吐/指标
- **问题与结论**：遇到的坑/原因/解决
- **明日计划**：阻塞点与跟进

---

## 常见坑与规避

- **模板不一致**：训练与推理的 `template` 必须一致。
- **tokenizer 版本/remote code**：部分模型需 `trust_remote_code=True` 与较新 `transformers`。
- **特殊 token 丢失**：`parse_template` 依赖 `tokenizer.chat_template`，注意模型版本与权重来源。
- **量化与 LoRA 组合**：低显存优先 QLoRA；导出/合并注意开关与目标 dtype。
- **vLLM 差异**：与 HF 生成参数行为略有不同，关注 `stop_words/eos` 与张量并发。

---

## 速查命令（常用）

```bash
cd <PROJECT_ROOT>
llamafactory-cli help | cat
rg "register_template\(" src/llamafactory/data/template.py | head -n 50 | cat
python -c "from transformers import AutoTokenizer as T;print(T.from_pretrained('qwen/Qwen1.5-0.5B-Chat',trust_remote_code=True).chat_template[:400])"
```

---

## 建议笔记结构

- `notes/dayX.md`：每日阅读与实验记录
- `notes/templates.md`：各模板渲染差异词典（角色标记/停止符/思维标签/工具调用）
- `notes/training_recipes.md`：SFT/DPO/RM/PPO/KTO/PT 配方与经验
- `notes/infer_backends.md`：HF/vLLM/SGLang 对比结论与推荐设置

---

## 📝 文档说明

**本文档由 [Cursor](https://cursor.sh/) AI 代码编辑器生成**

- **生成目的**：为 LLaMA-Factory 学习者提供系统化的 7 天学习路线图
- **内容特点**：结合理论学习和动手实践，包含详细的代码示例和操作步骤
- **适用对象**：希望深入理解 LLaMA-Factory 架构和具备二次开发能力的开发者
- **使用方法**：按照 Day 1-7 的顺序逐步学习，每个阶段都有明确的学习目标和验收标准

> 💡 **提示**：建议在学习过程中结合官方文档和源码阅读，以获得更全面的理解。



