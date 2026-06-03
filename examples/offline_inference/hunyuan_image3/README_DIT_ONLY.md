# 混元Image DiT-only IT2I 使用指南

## 概述

该脚本实现了混元Image的纯DiT架构进行image-to-image推理，不依赖AR阶段。

## 快速开始

### 基础用法

```bash
cd /d/huawei/code/claude_code/vllm_omni/vllm-omni

python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "将猫改成狗" \
  --image-path /path/to/your/input.jpg \
  --output ./dit_only_results \
  --steps 50 \
  --guidance-scale 5.0 \
  --seed 42
```

### 主要参数说明

- `--model`: 模型路径（默认: tencent/HunyuanImage-3.0-Instruct）
- `--prompts`: 文本prompt（必需，可以指定多个）
- `--image-path`: 输入图片路径（必需，多张图用逗号分隔，最多3张）
- `--output`: 输出目录（默认: 当前目录）
- `--steps`: 推理步数（默认: 50）
- `--guidance-scale`: CFG guidance scale（默认: 5.0）
- `--seed`: 随机种子（默认: 42）
- `--height/--width`: 输出尺寸（可选，默认使用输入图片尺寸）
- `--deploy-config`: 自定义配置文件（默认: hunyuan_image3_dit.yaml）

### 多图输入示例

```bash
python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "融合这些猫咪的特征，生成新的猫咪形象" \
  --image-path "cat1.jpg,cat2.jpg,cat3.jpg" \
  --output ./results \
  --steps 50 \
  --guidance-scale 5.0
```

### 指定输出尺寸

```bash
python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "改变图片风格为水彩画" \
  --image-path input.jpg \
  --output ./results \
  --height 1024 \
  --width 1024 \
  --steps 50 \
  --guidance-scale 5.0
```

## 核心设计

### DiT-only模式的关键实现

1. **配置文件**: 使用 `hunyuan_image3_dit.yaml`，只包含单个DiT stage
2. **Prompt构建**: 使用 `bot_task=None`，不添加AR相关的trigger tag（如`<think>`）
3. **输入处理**: 图片通过 `multi_modal_data.image` 传入，自动编码为VAE+ViT特征
4. **尺寸控制**: 直接指定height/width，不依赖AR的ratio token

### 与AR+DiT模式的区别

| 特性 | AR+DiT模式 | DiT-only模式 |
|------|-----------|-------------|
| 阶段数 | 2 (AR → DiT) | 1 (仅DiT) |
| CoT文本 | AR生成 | 无 |
| Prompt格式 | `bot_task="think"`，包含`<think>`标签 | `bot_task=None`，纯文本 |
| 尺寸预测 | AR预测ratio token | 直接指定或使用输入尺寸 |
| 推理速度 | 较慢（两阶段） | 较快（单阶段） |
| 生成质量 | 更好（有CoT增强） | 略低（无CoT） |

## 配置调整

### 单卡运行

如果只有1张GPU，需要修改 `vllm_omni/deploy/hunyuan_image3_dit.yaml`:

```yaml
stages:
  - stage_id: 0
    devices: "0"  # 改为单卡
    parallel_config:
      tensor_parallel_size: 1  # 改为1
      enable_expert_parallel: false  # 关闭expert parallel
```

### NPU平台

配置文件已包含NPU适配，使用时指定platform:

```bash
# 配置文件中已有npu section，会自动使用
python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "你的prompt" \
  --image-path input.jpg \
  --output ./results
```

## 性能优化参数

### KV Cache优化

```bash
python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "你的prompt" \
  --image-path input.jpg \
  --output ./results \
  --diffusion-kv-cache-dtype fp8 \
  --diffusion-kv-cache-skip-steps "0,1" \
  --diffusion-kv-cache-skip-layers "0-2"
```

### VAE Tiling（节省显存）

```bash
python examples/offline_inference/hunyuan_image3/dit_only_it2i.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompts "你的prompt" \
  --image-path large_image.jpg \
  --output ./results \
  --vae-use-tiling
```

## 对比测试

### 运行AR+DiT模式（对比）

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_moe.yaml \
  --modality img2img \
  --prompts "将猫改成狗" \
  --image-path input.jpg \
  --output ./ar_dit_results \
  --steps 50 \
  --guidance-scale 5.0 \
  --seed 42
```

## 故障排查

### 问题1: DiT初始化失败

**现象**: 报错 "CUDA out of memory" 或 "tensor parallel size mismatch"

**解决**:
- 检查GPU数量是否与 `tensor_parallel_size` 匹配
- 单卡运行需修改配置文件（见上述"单卡运行"）

### 问题2: 图片尺寸报错

**现象**: 报错尺寸不匹配或超出范围

**解决**:
- 确保height和width都是64的倍数
- 不指定尺寸时会自动使用输入图片尺寸

### 问题3: 生成质量不佳

**可能原因**: DiT-only缺少AR的CoT增强

**优化方向**:
- 使用更详细的prompt描述
- 调整guidance_scale（推荐范围: 3.0-7.0）
- 增加推理步数（如steps=100）

## 代码架构

```
dit_only_it2i.py
├── parse_args()                    # 解析命令行参数
├── main()
│   ├── Omni初始化                  # mode="image-editing"
│   ├── 加载图片                    # PIL.Image.open()
│   ├── Tokenizer初始化             # AutoTokenizer
│   ├── build_prompt_tokens()      # bot_task=None
│   ├── 构建formatted_prompts       # 包含token_ids, image, height/width
│   ├── 配置采样参数                # OmniDiffusionSamplingParams
│   └── omni.generate()            # 执行推理
└── 保存结果                        # output_*.png
```

## 参考文档

- 完整计划: `C:\Users\58479\.claude\plans\image-ar-dit-it2i-dit-it2i-ar-dit-dit-d-tidy-cerf.md`
- DiT Pipeline: `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
- Prompt工具: `vllm_omni/diffusion/models/hunyuan_image3/prompt_utils.py`
- 配置文件: `vllm_omni/deploy/hunyuan_image3_dit.yaml`
