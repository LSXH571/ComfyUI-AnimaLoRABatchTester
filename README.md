# ComfyUI-AnimaLoRABatchTester

Anima（Qwen / flow）底模专用的 LoRA 批量对比节点：一个节点内加载 UNET / CLIP / VAE，最多选 20 个 LoRA 逐张出图。同一批使用相同 seed、潜空间、提示词和采样参数，画面差异只来自 LoRA。

- 节点类名：`AnimaLoRABatchTester`
- 显示名：`Anima LoRA 批量测试`
- 分类：`Anima Batch`
- 输出：`IMAGE`

## 安装

1. 把整个文件夹放到 `ComfyUI/custom_nodes/` 下；
2. 完全重启 ComfyUI；
3. 浏览器 `Ctrl + F5` 强刷；
4. 画布双击搜索 `Anima LoRA 批量测试`。

无额外 pip 依赖，`torch` / `Pillow` / `comfy` 用 ComfyUI 自带环境。

## 模型准备

| 界面参数 | 说明 | 目录 |
|---|---|---|
| `unet_name` | Anima 底模 | `models/diffusion_models` |
| `clip_name` | Qwen 文本编码器 | `models/clip` |
| `vae_name` | Anima 配套 VAE | `models/vae` |

> `clip_type` 必须选 `stable_diffusion`。选 `qwen_image` 会出青绿红噪点乱码。

## 使用步骤

1. 添加节点，选好 UNET / CLIP / VAE；
2. 在**正面提示词**、**负面提示词**框填写内容（可纵向拉伸）；
3. `lora_count` 填要测的数量（1–20，默认 10），在展开的槽位里选 LoRA，`None` 表示跳过；
4. 采样参数保持默认即可：`euler` / `simple` / 30 步 / `cfg 4.0` / 1024×1536，固定 `seed`；
5. 运行。`include_none` 开启时会先出一张无 LoRA 基准图。

## 参数一览

| 分组 | 参数 | 默认 / 范围 |
|---|---|---|
| 模型 | `unet_name` / `clip_name` / `vae_name` / `clip_type` | `stable_diffusion` |
| 提示词 | `正面提示词 positive` / `负面提示词 negative` | 空，多行 |
| LoRA | `lora_count` | 1–20，默认 10 |
| LoRA | `lora_1` … `lora_20` | `None` 或本机 LoRA 文件 |
| LoRA | `extra_loras` | 每行一个，支持 `文件`、`文件:模型强度`、`文件:模型强度:CLIP强度`、`目录/` |
| 强度 | `default_model_strength` / `default_clip_strength` | 0.8 / 1.0（0–2） |
| 基准图 | `include_none` | 开 |
| 采样 | `sampler` / `scheduler` | `euler` / `simple` |
| 采样 | `steps` / `cfg` | 30（1–200）/ 4.0（0–20） |
| 尺寸 | `width` / `height` | 1024 / 1536（64–4096，建议 16 的倍数） |
| 输出 | `output_prefix` / `image_format` / `seed` | `Anima-LoRA-batch` / `webp` / 0 |

## 输出

```text
output/<output_prefix>/images/img_<序号>_<LoRA名>.<格式>
output/<output_prefix>/manifest.json
```

- 每张图会直接在节点上显示缩略图；
- `manifest.json` 记录每张图的模型、提示词口径、采样参数和文件名；
- 重跑覆盖同名文件，换批次请改 `output_prefix`。

## 注意事项

- 新放进 `loras` 目录的文件，重启或刷新后才会出现在下拉里；
- 8 GB 显存建议先跑 3–5 个 LoRA，确认画面后再放量；
- 参数非法（NaN、越界、采样器 / 格式不合法）会在控制台给出中文报错，按提示回界面重选；
- 旧工作流参数错位时，前端会把采样参数组恢复为默认值并在浏览器控制台告警，提示词和 LoRA 选择不受影响。

## 已知限制

- 仅按 Anima / Qwen flow 口径验证；
- 20 槽满额未做显存压测；
- 20 个槽位共用一套默认强度，逐 LoRA 不同强度请用 `extra_loras`；
- 不合并 LoRA stack（每个 LoRA 独立出图）。

## 目录结构

```text
ComfyUI-AnimaLoRABatchTester/
├── __init__.py
├── anima_lora_batch_tester.py
├── web/anima_batch.js
├── examples/anima-lora-batch-test.json
├── README.md
├── LICENSE
└── requirements.txt
```
