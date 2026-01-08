# LLaMA Factory 模型微调完整流程

## 📚 一、框架概述

LLaMA Factory 是一个统一的大模型微调框架，具有以下特性：

### 支持的微调方法
- **LoRA**: 低秩适应，参数高效
- **QLoRA**: 量化LoRA，更省显存
- **全量微调**: Full Fine-tuning
- **冻结微调**: Freeze tuning

### 支持的训练阶段
- **PT** (Pre-training): 预训练
- **SFT** (Supervised Fine-Tuning): 监督微调
- **RM** (Reward Modeling): 奖励建模
- **PPO**: 近端策略优化
- **DPO**: 直接偏好优化
- **KTO**: Kahneman-Tversky优化

### 支持的模型
LLaMA、Qwen、ChatGLM、Baichuan、DeepSeek等100+模型

### 支持的数据格式
- Alpaca格式
- ShareGPT格式
- 多轮对话格式
- 工具调用格式

---

## 🗂️ 二、项目目录结构

```
02Agent微调智能客服/
├── LLaMA-Factory/              # LLaMA Factory 框架
│   └── data/                   # 数据集目录
│       ├── dataset_info.json   # 数据集配置文件(核心!)
│       ├── glaive_toolcall_zh_demo.json  # 工具调用中文示例
│       ├── glaive_toolcall_en_demo.json  # 工具调用英文示例
│       ├── alpaca_zh_demo.json          # Alpaca中文示例
│       └── ...                          # 其他数据集
├── code/                       # 训练配置文件目录
│   ├── qwen2_lora_sft.yaml     # Qwen2 LoRA微调配置
│   ├── qwen_lora_merged.yaml   # 模型合并配置
│   └── function_call.py        # 功能调用脚本
├── cache/checkpoints/          # 训练输出目录(自动生成)
│   └── agent/                  # 模型checkpoint
│       ├── checkpoint-1000/
│       ├── checkpoint-2000/
│       └── training_loss.png   # 损失曲线
└── agent.ipynb                 # Jupyter notebook
```

---

## 🔧 三、核心组件详解

### 3.1 数据集配置文件: dataset_info.json

这是整个微调流程的**数据中枢**，定义了所有可用的数据集。

#### 本地数据集配置示例
```json
{
  "glaive_toolcall_zh_demo": {
    "file_name": "glaive_toolcall_zh_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "tools": "tools"
    }
  },
  "alpaca_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  }
}
```

#### 在线数据集配置示例
```json
{
  "glaive_toolcall_zh": {
    "hf_hub_url": "llamafactory/glaive_toolcall_zh",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "tools": "tools"
    }
  },
  "alpaca_gpt4_zh": {
    "hf_hub_url": "llamafactory/alpaca_gpt4_zh",
    "ms_hub_url": "llamafactory/alpaca_gpt4_zh"
  }
}
```

**关键字段说明**:
- `file_name`: 本地文件名
- `hf_hub_url`: HuggingFace Hub地址
- `ms_hub_url`: ModelScope Hub地址
- `formatting`: 数据格式(alpaca/sharegpt)
- `columns`: 字段映射
- `subset`: 数据子集
- `split`: 数据分片

### 3.2 训练配置文件: qwen2_lora_sft.yaml

```yaml
### model - 模型配置
model_name_or_path: qwen/Qwen2-1.5B  # 基座模型路径或HF模型名

### method - 微调方法配置
stage: sft                            # 训练阶段: sft/pt/rm/ppo/dpo/kto
do_train: true                        # 是否训练
finetuning_type: lora                 # 微调类型: lora/freeze/full
lora_target: all                      # LoRA目标层: all/q_proj,v_proj/...
# lora_rank: 8                        # LoRA秩(可选)
# lora_alpha: 16                      # LoRA缩放系数(可选)

### dataset - 数据集配置
dataset_dir: LLaMA-Factory/data       # 数据集目录(dataset_info.json所在位置)
dataset: glaive_toolcall_en, glaive_toolcall_zh, alpaca_gpt4_en, alpaca_gpt4_zh
template: qwen                        # 对话模板: qwen/llama3/chatglm3/...
cutoff_len: 1024                      # 最大序列长度
max_samples: 50000                    # 最大样本数(用于快速测试)
overwrite_cache: true                 # 是否覆盖缓存
preprocessing_num_workers: 16         # 数据预处理进程数

### output - 输出配置
output_dir: ./cache/checkpoints/agent # 输出目录
logging_steps: 100                    # 日志打印间隔
save_steps: 1000                      # 模型保存间隔
plot_loss: true                       # 是否绘制损失曲线
overwrite_output_dir: true            # 是否覆盖输出目录

### train - 训练超参数
per_device_train_batch_size: 1        # 每个设备的批次大小
gradient_accumulation_steps: 8        # 梯度累积步数(有效batch=1*8=8)
learning_rate: 1.0e-4                 # 学习率
num_train_epochs: 3.0                 # 训练轮数
lr_scheduler_type: cosine             # 学习率调度器: cosine/linear/constant
warmup_ratio: 0.1                     # 预热比例
bf16: False                           # 是否使用BF16混合精度(需GPU支持)
# fp16: true                          # 是否使用FP16混合精度
ddp_timeout: 180000000                # 分布式训练超时时间

### eval - 验证配置
val_size: 0.01                        # 验证集比例
per_device_eval_batch_size: 1         # 验证批次大小
eval_strategy: steps                  # 评估策略: steps/epoch/no
eval_steps: 1000                      # 评估间隔步数
```

---

## 📊 四、数据格式详解

### 4.1 Alpaca格式 (指令微调)

```json
[
  {
    "instruction": "用户指令",
    "input": "可选的输入内容",
    "output": "期望的输出",
    "system": "可选的系统提示词",
    "history": [
      ["历史问题1", "历史回答1"],
      ["历史问题2", "历史回答2"]
    ]
  }
]
```

### 4.2 ShareGPT格式 (多轮对话)

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户问题"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ]
  }
]
```

### 4.3 工具调用格式 (Agent训练)

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "我需要为John Doe生成一张发票。他购买了2个苹果，每个$1，以及3根香蕉，每根$0.5。"
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"generate_invoice\", \"arguments\": {\"customer_name\": \"约翰·多伊\", \"items\": [{\"name\": \"苹果\", \"quantity\": 2, \"price\": 1}, {\"name\": \"香蕉\", \"quantity\": 3, \"price\": 0.5}]}}"
      },
      {
        "from": "observation",
        "value": "{\"invoice_id\": \"INV12345\", \"customer_name\": \"约翰·多伊\", \"items\": [{\"name\": \"苹果\", \"quantity\": 2, \"price\": 1, \"total\": 2}, {\"name\": \"香蕉\", \"quantity\": 3, \"price\": 0.5, \"total\": 1.5}], \"total\": 3.5, \"status\": \"生成\"}"
      },
      {
        "from": "gpt",
        "value": "发票已成功生成。发票编号为INV12345。约翰·多伊的总金额为$3.5。"
      }
    ],
    "tools": "[{\"name\": \"generate_invoice\", \"description\": \"生成发票\", \"parameters\": {\"type\": \"object\", \"properties\": {\"customer_name\": {\"type\": \"string\", \"description\": \"客户名称\"}, \"items\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}, \"quantity\": {\"type\": \"integer\"}, \"price\": {\"type\": \"number\"}}, \"required\": [\"name\", \"quantity\", \"price\"]}}}, \"required\": [\"customer_name\", \"items\"]}}]"
  }
]
```

**工具调用流程**:
```
User → 问题
  ↓
Model → function_call (调用工具)
  ↓
System → observation (工具返回结果)
  ↓
Model → 最终回答
```

---

## 🚀 五、完整微调流程(5步)

### 步骤1: 准备数据集

#### 方式A: 使用本地数据

1. 准备JSON格式的训练数据 `my_data.json`
2. 将文件放到 `LLaMA-Factory/data/` 目录
3. 在 `dataset_info.json` 中注册数据集:

```json
{
  "my_custom_dataset": {
    "file_name": "my_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

#### 方式B: 使用在线数据集

直接在 `dataset_info.json` 中配置HuggingFace Hub URL:

```json
{
  "alpaca_gpt4_zh": {
    "hf_hub_url": "llamafactory/alpaca_gpt4_zh"
  }
}
```

训练时会自动下载到缓存目录。

### 步骤2: 编写训练配置文件

创建 `my_config.yaml`:

```yaml
### model
model_name_or_path: qwen/Qwen2-1.5B

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset_dir: LLaMA-Factory/data
dataset: my_custom_dataset
template: qwen
cutoff_len: 1024

### output
output_dir: ./checkpoints/my_model
logging_steps: 10
save_steps: 100

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5e-5
num_train_epochs: 3
bf16: true

### eval
val_size: 0.1
eval_strategy: steps
eval_steps: 100
```

### 步骤3: 启动训练

#### 在Jupyter Notebook中:
```python
!llamafactory-cli train ./code/my_config.yaml
```

#### 在命令行中:
```bash
# Windows
cd "c:\Users\wxb55\Desktop\LLM_LearningRecord\练习玩具\02Agent微调智能客服"
llamafactory-cli train ./code/qwen2_lora_sft.yaml

# Linux/Mac
cd /path/to/project
llamafactory-cli train ./code/qwen2_lora_sft.yaml
```

#### 使用Python脚本:
```python
from llamafactory.train.tuner import run_exp

args = {
    "model_name_or_path": "qwen/Qwen2-1.5B",
    "stage": "sft",
    "finetuning_type": "lora",
    "dataset": "my_custom_dataset",
    "output_dir": "./checkpoints/my_model",
    # ... 其他参数
}

run_exp(args)
```

### 步骤4: 监控训练过程

#### 训练日志输出示例:
```
***** Running training *****
  Num examples = 10000
  Num Epochs = 3
  Instantaneous batch size per device = 1
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 8
  Total optimization steps = 3750

Step 100/3750: loss=2.345, learning_rate=9.5e-5, epoch=0.08
Step 200/3750: loss=1.987, learning_rate=9.0e-5, epoch=0.16
...
Saving model checkpoint to ./checkpoints/agent/checkpoint-1000
...
```

#### 检查输出目录:
```
checkpoints/agent/
├── checkpoint-1000/
│   ├── adapter_config.json      # LoRA配置
│   ├── adapter_model.safetensors # LoRA权重(主要文件)
│   ├── trainer_state.json       # 训练状态
│   ├── training_args.bin        # 训练参数
│   └── optimizer.pt             # 优化器状态
├── checkpoint-2000/
├── checkpoint-3000/
└── training_loss.png            # 损失曲线图
```

#### 查看损失曲线:
训练完成后会生成 `training_loss.png`，可以直观看到:
- 训练损失下降趋势
- 验证损失变化
- 是否过拟合

### 步骤5: 模型导出与推理

#### 选项A: 直接使用LoRA适配器(推荐)

**优点**: 节省磁盘空间，只需保存LoRA权重(通常几百MB)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2-1.5B",
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/agent/checkpoint-3000"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    "qwen/Qwen2-1.5B",
    trust_remote_code=True
)

# 推理
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

#### 选项B: 合并为完整模型

**优点**: 推理速度更快，便于部署

创建合并配置文件 `qwen_lora_merged.yaml`:

```yaml
### model
model_name_or_path: qwen/Qwen2-1.5B
adapter_name_or_path: checkpoints/agent/checkpoint-3000
template: qwen
finetuning_type: lora

### export
export_dir: models/qwen2-agent-merged
export_size: 2                    # 分片大小(GB)
export_device: cpu                # 导出设备
export_legacy_format: false       # 是否使用旧格式
```

执行合并:
```bash
llamafactory-cli export qwen_lora_merged.yaml
```

合并后的模型可直接使用:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/qwen2-agent-merged",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/qwen2-agent-merged",
    trust_remote_code=True
)

# 推理
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## 🔍 六、高级配置与优化

### 6.1 LoRA参数调优

```yaml
finetuning_type: lora
lora_rank: 8                      # LoRA秩(默认8)，越大越接近全量微调
lora_alpha: 16                    # 缩放系数(默认16)，通常设为rank的2倍
lora_dropout: 0.1                 # Dropout率
lora_target: all                  # 目标模块
# lora_target: q_proj,v_proj     # 也可指定具体层
```

**参数影响**:
- `lora_rank` ↑ → 参数量↑、效果↑、显存↑
- `lora_alpha` ↑ → LoRA权重影响增强
- `lora_target: all` → 所有线性层(推荐)

### 6.2 量化训练(QLoRA)

**节省50-70%显存**:

```yaml
finetuning_type: lora
quantization_bit: 4               # 量化位数: 4/8
quantization_type: nf4            # 量化类型: nf4/fp4
double_quantization: true         # 双重量化(进一步节省显存)
```

### 6.3 混合精度训练

```yaml
# 方式1: BF16(推荐，需Ampere架构GPU)
bf16: true

# 方式2: FP16
fp16: true

# 方式3: 纯FP32(最慢但最稳定)
bf16: false
fp16: false
```

### 6.4 梯度检查点(显存优化)

```yaml
gradient_checkpointing: true      # 用计算换显存
```

### 6.5 DeepSpeed优化(多卡训练)

创建 `ds_config.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

在配置文件中引用:
```yaml
deepspeed: ds_config.json
```

### 6.6 学习率调度策略

```yaml
lr_scheduler_type: cosine         # cosine/linear/constant/cosine_with_restarts
warmup_ratio: 0.1                 # 预热比例
# warmup_steps: 500               # 或直接指定预热步数
```

### 6.7 数据增强

```yaml
# 数据打乱
shuffle: true

# 数据重复
num_train_epochs: 3

# 序列长度优化
cutoff_len: 1024
packing: false                    # 是否打包短序列
```

---

## 🐛 七、常见问题与解决方案

### 问题1: 数据集找不到

**错误**: `ValueError: Cannot open data\dataset_info.json`

**原因**:
- `dataset_info.json` 路径不正确
- 未指定 `dataset_dir` 参数

**解决**:
```yaml
dataset_dir: LLaMA-Factory/data   # 添加此行
dataset: your_dataset_name
```

---

### 问题2: 显存不足(OOM)

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**(按优先级):

#### 方案1: 减小批次大小
```yaml
per_device_train_batch_size: 1    # 从2改为1
gradient_accumulation_steps: 16   # 从8改为16(保持有效batch不变)
```

#### 方案2: 启用梯度检查点
```yaml
gradient_checkpointing: true
```

#### 方案3: 使用量化(QLoRA)
```yaml
quantization_bit: 4
quantization_type: nf4
```

#### 方案4: 减小序列长度
```yaml
cutoff_len: 512                   # 从1024改为512
```

#### 方案5: 减小LoRA秩
```yaml
lora_rank: 4                      # 从8改为4
```

---

### 问题3: 训练速度慢

**解决方案**:

#### 方案1: 启用混合精度
```yaml
bf16: true                        # 或 fp16: true
```

#### 方案2: 增加批次大小
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

#### 方案3: 减少数据预处理开销
```yaml
preprocessing_num_workers: 8      # 增加预处理进程
overwrite_cache: false            # 使用缓存数据
```

#### 方案4: 限制训练样本数(快速测试)
```yaml
max_samples: 1000                 # 只使用1000个样本
```

---

### 问题4: 模型效果不佳

**诊断与解决**:

#### 症状1: Loss不下降

**可能原因**:
- 学习率过大或过小
- 数据质量问题
- 模板不匹配

**解决**:
```yaml
# 调整学习率
learning_rate: 5e-5               # 尝试 1e-5 ~ 1e-4

# 使用正确的模板
template: qwen                    # 确保与模型匹配

# 检查数据
# 打印前几个样本确认格式正确
```

#### 症状2: Loss下降但效果差

**可能原因**:
- 训练不充分
- 过拟合
- 验证集与训练集分布不一致

**解决**:
```yaml
# 增加训练轮数
num_train_epochs: 5.0

# 增加训练数据
max_samples: 100000

# 调整验证集比例
val_size: 0.05
```

#### 症状3: 训练集好但验证集差(过拟合)

**解决**:
```yaml
# 增加Dropout
lora_dropout: 0.1

# 减少训练轮数
num_train_epochs: 2.0

# 使用正则化
weight_decay: 0.01
```

---

### 问题5: 模板不匹配

**错误**: 模型输出格式混乱

**原因**: 使用了错误的对话模板

**常见模板**:
```yaml
template: qwen                    # Qwen系列
template: llama3                  # LLaMA 3系列
template: chatglm3                # ChatGLM3系列
template: baichuan2               # Baichuan2系列
template: default                 # 通用模板
```

**查看支持的模板**:
```bash
llamafactory-cli train --help
```

---

### 问题6: 多卡训练问题

**单卡训练**:
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config.yaml
```

**多卡训练(DDP)**:
```bash
# 方式1: torchrun
torchrun --nproc_per_node 4 -m llamafactory.train config.yaml

# 方式2: accelerate
accelerate launch -m llamafactory.train config.yaml
```

**DeepSpeed多卡**:
```bash
deepspeed --num_gpus 4 -m llamafactory.train config.yaml
```

---

### 问题7: 推理时输出异常

**问题**: 微调后模型输出乱码或格式错误

**原因**:
- 未使用与训练相同的模板
- 推理参数不当

**解决**:
```python
# 使用正确的生成参数
outputs = model.generate(
    **inputs,
    max_new_tokens=512,           # 最大生成长度
    temperature=0.7,              # 温度(0.1-1.0)
    top_p=0.9,                    # 核采样
    top_k=50,                     # Top-K采样
    repetition_penalty=1.1,       # 重复惩罚
    do_sample=True,               # 启用采样
    pad_token_id=tokenizer.eos_token_id
)
```

---

## 📈 八、模型评估

### 8.1 使用验证集评估

在训练配置中启用评估:

```yaml
### eval
val_size: 0.1                     # 10%作为验证集
eval_strategy: steps              # 按步数评估
eval_steps: 500                   # 每500步评估一次
per_device_eval_batch_size: 2
```

训练日志中会显示:
```
Step 500: train_loss=1.234, eval_loss=1.456
Step 1000: train_loss=0.987, eval_loss=1.123
```

### 8.2 使用测试脚本评估

创建 `evaluate.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2-1.5B")
model = PeftModel.from_pretrained(base_model, "checkpoints/agent/checkpoint-3000")
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-1.5B")

# 加载测试数据
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 评估
correct = 0
total = len(test_data)

for item in test_data:
    prompt = item["instruction"]
    expected = item["output"]

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 简单的准确率评估
    if expected in prediction:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
```

### 8.3 使用LLaMA Factory内置评估

创建评估配置 `eval_config.yaml`:

```yaml
model_name_or_path: qwen/Qwen2-1.5B
adapter_name_or_path: checkpoints/agent/checkpoint-3000
finetuning_type: lora
template: qwen

task: mmlu                        # 评估任务
split: test
lang: zh                          # 语言

output_dir: eval_results
per_device_eval_batch_size: 4
```

运行评估:
```bash
llamafactory-cli eval eval_config.yaml
```

---

## 🎯 九、实战案例

### 案例1: 智能客服Agent微调

**目标**: 训练模型具备工具调用能力，实现智能客服功能

**数据准备**:
```json
[
  {
    "conversations": [
      {"from": "human", "value": "帮我查询订单123456的物流信息"},
      {"from": "function_call", "value": "{\"name\": \"query_logistics\", \"arguments\": {\"order_id\": \"123456\"}}"},
      {"from": "observation", "value": "{\"status\": \"运输中\", \"location\": \"北京\", \"estimated_arrival\": \"2024-01-15\"}"},
      {"from": "gpt", "value": "您的订单123456正在运输中，当前位置在北京，预计2024-01-15送达。"}
    ],
    "tools": "[{\"name\": \"query_logistics\", \"description\": \"查询物流信息\", \"parameters\": {...}}]"
  }
]
```

**配置文件**:
```yaml
model_name_or_path: qwen/Qwen2-7B
stage: sft
finetuning_type: lora
lora_target: all

dataset_dir: data
dataset: customer_service_toolcall
template: qwen
cutoff_len: 2048

output_dir: checkpoints/customer_service_agent
num_train_epochs: 3
learning_rate: 1e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### 案例2: 指令微调(Instruction Tuning)

**目标**: 让模型更好地理解和执行指令

**数据准备** (Alpaca格式):
```json
[
  {
    "instruction": "将以下文本翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is very nice today."
  },
  {
    "instruction": "写一首关于春天的诗",
    "input": "",
    "output": "春风拂面暖如酥，\n万物复苏展新图。\n..."
  }
]
```

**配置文件**:
```yaml
model_name_or_path: qwen/Qwen2-1.5B
stage: sft
finetuning_type: lora

dataset: alpaca_zh_custom
template: qwen
cutoff_len: 512

num_train_epochs: 5
learning_rate: 5e-5
```

### 案例3: DPO偏好对齐

**目标**: 通过人类偏好数据对齐模型输出

**数据准备**:
```json
[
  {
    "conversations": [
      {"from": "human", "value": "请介绍一下人工智能"}
    ],
    "chosen": [
      {"from": "gpt", "value": "人工智能(AI)是计算机科学的一个分支..."}
    ],
    "rejected": [
      {"from": "gpt", "value": "AI就是机器学习。"}
    ]
  }
]
```

**配置文件**:
```yaml
model_name_or_path: checkpoints/customer_service_agent  # 使用SFT后的模型
stage: dpo
finetuning_type: lora

dataset: preference_data
template: qwen

dpo_beta: 0.1                     # DPO温度参数
learning_rate: 5e-6               # DPO通常用更小的学习率
num_train_epochs: 1
```

---

## 📚 十、最佳实践建议

### 10.1 数据质量 > 数据数量

- 优先保证数据质量
- 清洗错误标注
- 保持数据格式一致
- 多样性 > 重复性

### 10.2 渐进式训练策略

```
阶段1: 小数据集快速验证
  ↓
阶段2: 全数据集初步训练
  ↓
阶段3: 超参数调优
  ↓
阶段4: 多数据集混合训练
```

### 10.3 超参数推荐值

**小模型(1-3B)**:
```yaml
learning_rate: 1e-4
lora_rank: 8
per_device_train_batch_size: 4
num_train_epochs: 3-5
```

**中型模型(7-13B)**:
```yaml
learning_rate: 5e-5
lora_rank: 8-16
per_device_train_batch_size: 2
num_train_epochs: 2-3
```

**大型模型(30B+)**:
```yaml
learning_rate: 1e-5
lora_rank: 16-32
per_device_train_batch_size: 1
num_train_epochs: 1-2
quantization_bit: 4               # 建议使用QLoRA
```

### 10.4 监控关键指标

训练过程中重点关注:
- **Loss曲线**: 应平稳下降
- **Learning Rate**: 查看调度是否正常
- **显存占用**: 避免OOM
- **训练速度**: samples/second

### 10.5 版本管理

```
checkpoints/
├── experiment1_baseline/
│   └── checkpoint-3000/
├── experiment2_larger_lr/
│   └── checkpoint-3000/
├── experiment3_more_data/
│   └── checkpoint-3000/
└── training_logs.txt
```

记录每次实验的:
- 配置参数
- 数据集版本
- 最终指标
- 问题与改进

---

## 🔗 十一、常用命令速查

### 训练相关
```bash
# 基础训练
llamafactory-cli train config.yaml

# 指定GPU
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config.yaml

# 从checkpoint恢复
llamafactory-cli train config.yaml --resume_from_checkpoint checkpoints/xxx

# 多卡训练
torchrun --nproc_per_node 4 -m llamafactory.train config.yaml
```

### 导出相关
```bash
# 导出LoRA合并模型
llamafactory-cli export export_config.yaml

# 导出量化模型
llamafactory-cli export export_config.yaml --export_quantization_bit 4
```

### 评估相关
```bash
# 评估模型
llamafactory-cli eval eval_config.yaml

# WebUI启动
llamafactory-cli webui
```

### 查看帮助
```bash
# 查看所有命令
llamafactory-cli --help

# 查看训练参数
llamafactory-cli train --help

# 查看支持的模型
llamafactory-cli list-models
```

---

## 🌟 十二、进阶资源

### 官方资源
- GitHub: https://github.com/hiyouga/LLaMA-Factory
- 文档: https://github.com/hiyouga/LLaMA-Factory/wiki
- 示例: `LLaMA-Factory/examples/`

### 数据集资源
- HuggingFace Hub: https://huggingface.co/datasets
- ModelScope: https://modelscope.cn/datasets
- LLaMA Factory内置: 100+ 数据集

### 学习路径
1. 先跑通官方demo
2. 使用小数据集(100-1000样本)快速迭代
3. 逐步扩大数据规模
4. 尝试不同微调策略(LoRA → DPO → PPO)
5. 部署到生产环境

---

## 📝 十三、常见配置模板

### 模板1: 快速原型验证

```yaml
# 用于快速测试，几分钟完成
model_name_or_path: qwen/Qwen2-1.5B
stage: sft
finetuning_type: lora
lora_target: all

dataset: your_dataset
template: qwen
cutoff_len: 512
max_samples: 100                  # 只用100个样本

output_dir: ./quick_test
per_device_train_batch_size: 2
num_train_epochs: 1
learning_rate: 1e-4
bf16: true
```

### 模板2: 生产级训练

```yaml
model_name_or_path: qwen/Qwen2-7B
stage: sft
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

dataset_dir: data
dataset: train_set1, train_set2, train_set3
template: qwen
cutoff_len: 2048

output_dir: ./production_model
logging_steps: 10
save_steps: 500
save_total_limit: 3               # 只保留最近3个checkpoint
plot_loss: true

per_device_train_batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
learning_rate: 5e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

val_size: 0.05
eval_strategy: steps
eval_steps: 500
per_device_eval_batch_size: 4
```

### 模板3: 低显存配置(8GB)

```yaml
model_name_or_path: qwen/Qwen2-1.5B
stage: sft
finetuning_type: lora
lora_rank: 4                      # 降低秩
quantization_bit: 4               # 4bit量化

dataset: your_dataset
template: qwen
cutoff_len: 512                   # 较短序列

output_dir: ./low_mem_model
per_device_train_batch_size: 1   # 批次1
gradient_accumulation_steps: 16  # 累积16步
gradient_checkpointing: true     # 启用检查点
learning_rate: 1e-4
num_train_epochs: 3
```

---

## ✅ 总结

LLaMA Factory 是一个强大而灵活的大模型微调框架。掌握以下核心要点:

1. **数据为王**: 高质量数据 > 复杂算法
2. **循序渐进**: 从小规模开始，逐步优化
3. **监控调试**: 密切关注训练指标
4. **合理配置**: 根据硬件条件选择合适参数
5. **版本管理**: 记录每次实验，便于复现

从快速原型到生产部署，LLaMA Factory 都能提供完整的解决方案！
