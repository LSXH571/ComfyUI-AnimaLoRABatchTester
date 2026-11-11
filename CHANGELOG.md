# Changelog

## v1.0.0

首个公开版本。

- 新增：Anima（Qwen / flow）底模的 LoRA 批量对比节点 `AnimaLoRABatchTester`
- 新增：节点内 20 个 LoRA 下拉槽位，`lora_count` 控制展开数量（默认 10）
- 新增：正面 / 负面提示词多行文本框，可纵向拉伸，两框之间保留 12px 间距
- 新增：`include_none` 输出无 LoRA 基准图
- 新增：逐 LoRA 独立打补丁、独立采样、独立存图，并写 `manifest.json`
- 新增：节点内直接显示已保存图片的缩略图（`ui.images`）
- 新增：`extra_loras` 支持行内模型强度 / CLIP 强度与整目录展开（自动去重）
- 新增：参数后端校验（NaN、越界、非法 sampler / scheduler / image_format 给出中文报错）
- 新增：前端参数自检，检测到旧工作流参数错位时恢复采样参数组默认值并告警
- 修复：旧工作流 `widgets_values` 项数不一致导致参数整体错位
- 修复：Qwen / Wan 视频式 VAE 返回 5 维张量时解码失败（统一为 BHWC）
- 修复：返回字典缺少 `result` 导致 outputs 为空、缓存回填 IndexError
- 修复：拖动提示词框后与后续控件重叠
- 修复：打开节点时提示词框自动无限增高
- 修复：`onNodeCreated` 阶段 `node.arrange()` 抛 `NullGraphError`