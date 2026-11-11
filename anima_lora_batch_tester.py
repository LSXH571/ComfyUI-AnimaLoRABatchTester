# -*- coding: utf-8 -*-
"""
Anima LoRA 批量测试节点（原生 ComfyUI 链路，绕开 easy-use / USCG Builder 的编码器与合并栈问题）。

链路（与本机已跑通的冒烟链路一致）：
  UNET 加载 -> CLIP(stable_diffusion) -> VAE -> 逐 LoRA 补丁 -> 编码正负提示词
  -> common_ksampler(euler/simple/30/cfg4/1.0) -> VAE 解码 -> 落盘 + 返回 IMAGE 批次。

每个选中的 LoRA 独立一条配置、固定同一个 seed、同一张潜空间，保证可横向对比。
"""
import os
import json
import time

import torch
from PIL import Image

import folder_paths
import comfy
import comfy.sd
import comfy.utils
from nodes import common_ksampler

# 下拉槽位数量上限：每个槽位可从本机 loras 列表里直接选一个，None 表示跳过该槽位。
LORA_SLOTS = 20


def _resolve_clip_type(name):
    """把界面 clip_type 字符串映射到 comfy.sd.CLIPType；未知值回退 stable_diffusion（Anima 口径）。"""
    t = getattr(comfy.sd.CLIPType, str(name).strip().upper(), None)
    return t if t is not None else comfy.sd.CLIPType.STABLE_DIFFUSION


def _lora_options():
    """下拉选项：None + 本机 loras 目录全部文件。"""
    return ["None"] + folder_paths.get_filename_list("loras")


def _encode(clip, text):
    """与 ComfyUI CLIPTextEncode 一致：tokenize -> encode_from_tokens -> [[cond, pooled]]。"""
    tokens = clip.tokenize(text)
    pooled = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = pooled.pop("cond")
    return [[cond, pooled]]


def _parse_lora_line(line, default_ms, default_cs):
    """
    高级文本框一行定义（可选）：
      文件名.safetensors
      文件名.safetensors:模型强度
      文件名.safetensors:模型强度:CLIP强度
      子目录/   -> 展开该目录下全部 LoRA（逐个测，不堆叠）
    返回 (rel_path, model_strength, clip_strength) 或 None。
    """
    line = (line or "").strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(":")]
    path = parts[0]
    ms = float(parts[1]) if len(parts) > 1 and parts[1] else default_ms
    cs = float(parts[2]) if len(parts) > 2 and parts[2] else default_cs
    return path, ms, cs


def _finite_number(value, name, minimum, maximum, cast=float):
    """Validate numeric inputs before they reach ComfyUI's sampler."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} 必须是有效数字，当前值: {value!r}")
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"{name} 不能是 NaN 或无穷大")
    if number < minimum or number > maximum:
        raise ValueError(f"{name} 超出范围 [{minimum}, {maximum}]：{number}")
    return cast(number)


def _safe_output_prefix(value):
    """Keep output paths inside ComfyUI output directory."""
    value = str(value or "Anima-LoRA-batch").strip()
    value = value.replace("\\\\", "/").replace("\\", "/").strip("/")
    parts = [p for p in value.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "Anima-LoRA-batch"


def _expand_lora_folder(prefix, default_ms, default_cs):
    """folder/ 语法：列出 loras 目录下前缀匹配的文件，逐个作为独立 LoRA 返回。"""
    prefix = prefix.rstrip("/")
    rels = []
    for f in folder_paths.get_filename_list("loras"):
        norm = f.replace("\\", "/")
        if norm == prefix or norm.startswith(prefix + "/"):
            rels.append(f)
    return [(r, default_ms, default_cs) for r in sorted(rels)]


class AnimaLoRABatchTester:
    """单节点完成：加载底模/编码器/VAE，循环逐 LoRA 采样、解码、存图。"""

    @classmethod
    def INPUT_TYPES(cls):
        lora_slots = {}
        for i in range(1, LORA_SLOTS + 1):
            lora_slots[f"lora_{i}"] = (_lora_options(), {"default": "None"})

        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                # Anima/Qwen 必须保持 stable_diffusion；qwen_image 是错误口径（出乱码）。
                "clip_type": (["stable_diffusion", "sdxl", "flux", "sd3", "qwen_image"], {"default": "stable_diffusion"}),
                "正面提示词 positive": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "正面提示词：角色、画风、构图、镜头、光影等正向描述",
                }),
                "负面提示词 negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "负面提示词：低质、坏手、多余肢体、水印、噪点等需要排除的内容",
                }),
                # 本次要测试的 LoRA 数量（1-20）；界面只展开对应数量的下拉槽位，默认 10。
                "lora_count": ("INT", {"default": 10, "min": 1, "max": LORA_SLOTS, "step": 1}),
                **lora_slots,
                "extra_loras": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "附加 LoRA（文本，可选）：每行一个 LoRA，支持 文件 / 文件:模型强度 / 文件:模型强度:CLIP强度 / 目录/",
                }),
                "default_model_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "default_clip_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "include_none": ("BOOLEAN", {"default": True}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 4096}),
                "output_prefix": ("STRING", {"default": "Anima-LoRA-batch"}),
                "image_format": (["webp", "png", "jpg"], {"default": "webp"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 63 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Anima Batch"
    OUTPUT_NODE = True

    # ---------- 模型加载 ----------
    @staticmethod
    def _load_unet(name):
        path = folder_paths.get_full_path("diffusion_models", name) or folder_paths.get_full_path("unet", name)
        if not path:
            raise FileNotFoundError(f"扩散模型未找到: {name}")
        print(f"[AnimaBatch] 加载 UNET: {name}")
        return comfy.sd.load_diffusion_model(path, model_options={})

    @staticmethod
    def _load_clip(name, clip_type_str):
        path = folder_paths.get_full_path("clip", name) or folder_paths.get_full_path("text_encoders", name)
        if not path:
            raise FileNotFoundError(f"文本编码器未找到: {name}")
        emb_dir = folder_paths.get_folder_paths("embeddings")
        print(f"[AnimaBatch] 加载 CLIP: {name} ({clip_type_str})")
        return comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=emb_dir,
            clip_type=_resolve_clip_type(clip_type_str),
        )

    @staticmethod
    def _load_vae(name):
        path = folder_paths.get_full_path("vae", name)
        if not path:
            raise FileNotFoundError(f"VAE 未找到: {name}")
        print(f"[AnimaBatch] 加载 VAE: {name}")
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    @staticmethod
    def _latent_spec(model, width, height):
        """按模型 latent_format 取通道数与下采样比例，避免 Qwen/flow 模型用错 EmptySD3/4 通道。"""
        lf = None
        try:
            lf = model.get_model_object("latent_format")
        except Exception:
            inner = getattr(model, "model", None)
            lf = getattr(inner, "latent_format", None)
            if lf is None and hasattr(inner, "model_config"):
                lf = getattr(inner.model_config, "latent_format", None)
        channels = getattr(lf, "latent_channels", 4)
        ratio = getattr(lf, "spacial_downscale_ratio", 8) or 8
        latent = {
            "samples": torch.zeros((1, channels, height // ratio, width // ratio)),
            "downscale_ratio_spacial": ratio,
        }
        return latent

    def _collect_lora_entries(self, slot_names, extra_loras, default_ms, default_cs):
        """下拉槽位 + 高级文本框合并，去重保持顺序。"""
        entries = []
        for name in slot_names:
            if name and name != "None":
                entries.append((name, default_ms, default_cs))
        for raw in (extra_loras or "").splitlines():
            parsed = _parse_lora_line(raw, default_ms, default_cs)
            if parsed is None:
                continue
            path, ms, cs = parsed
            ms = _finite_number(ms, f"{path} 模型强度", 0.0, 2.0)
            cs = _finite_number(cs, f"{path} CLIP 强度", 0.0, 2.0)
            if path.endswith("/"):
                entries.extend(_expand_lora_folder(path, ms, cs))
            else:
                entries.append((path, ms, cs))

        seen, dedup = set(), []
        for e in entries:
            key = e[0].replace("\\", "/")
            if key not in seen:
                seen.add(key)
                dedup.append(e)
        return dedup

    @staticmethod
    def _to_image_bhwc(decoded_tensor):
        """把 VAE.decode 的输出统一成 ComfyUI 的 IMAGE 形状 BHWC（B,H,W,C）。

        ComfyUI 的 VAE.decode 末尾只做 movedim(1, -1)；遇到 Wan/Qwen 这类视频式
        VAE 时会保留时间维，返回 B,T,H,W,C 五维（单图 T=1）。官方 VAEDecode 节点
        用 reshape 合并前两维，这里保持同样口径；同时兼容旧的 BCHW 四维输出。
        """
        image = torch.as_tensor(decoded_tensor).detach()
        if image.ndim == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        elif image.ndim == 4 and image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = image.movedim(1, -1)
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"VAE 解码结果维度异常: {tuple(decoded_tensor.shape)}")
        return image

    @staticmethod
    def _decode_pil(image_bhwc):
        """已归一化的 IMAGE(BHWC) -> PIL，取批次首张。"""
        arr = image_bhwc[0].detach().cpu().float().numpy()
        arr = (arr.clip(0.0, 1.0) * 255.0).round().astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr)

    def run(self, unet_name, clip_name, vae_name, clip_type, lora_count, extra_loras,
            default_model_strength, default_clip_strength, include_none,
            sampler, scheduler, steps, cfg, width, height,
            output_prefix, image_format, seed, **rest):

        positive = str(rest.get("正面提示词 positive", "") or "")
        negative = str(rest.get("负面提示词 negative", "") or "")

        # Reject stale/misaligned workflow values before they reach the sampler.
        default_model_strength = _finite_number(default_model_strength, "默认模型强度", 0.0, 2.0)
        default_clip_strength = _finite_number(default_clip_strength, "默认 CLIP 强度", 0.0, 2.0)
        steps = _finite_number(steps, "steps", 1, 200, int)
        cfg = _finite_number(cfg, "cfg", 0.0, 20.0)
        width = _finite_number(width, "width", 64, 4096, int)
        height = _finite_number(height, "height", 64, 4096, int)
        # JavaScript widgets use IEEE-754 numbers; stay within the exact integer range.
        seed = _finite_number(seed, "seed", 0, 2 ** 53 - 1, int)
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f"无效 sampler：{sampler!r}。请在节点中重新选择采样器。")
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(f"无效 scheduler：{scheduler!r}。请在节点中重新选择调度器。")
        if image_format not in ("png", "webp", "jpg"):
            raise ValueError(f"无效 image_format：{image_format!r}，仅支持 png/webp/jpg。")

        try:
            active = min(max(int(lora_count), 1), LORA_SLOTS)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"lora_count 必须为 1-{LORA_SLOTS} 的整数")
        slot_names = [rest.get(f"lora_{i}", "None") for i in range(1, active + 1)]
        entries = self._collect_lora_entries(slot_names, extra_loras, default_model_strength, default_clip_strength)
        if include_none:
            entries = [("None", None, None)] + entries
        if not entries:
            raise ValueError("未选择任何 LoRA，且未勾选 include_none；至少需要一个待测 LoRA 或基准图。")

        model = self._load_unet(unet_name)
        clip = self._load_clip(clip_name, clip_type)
        vae = self._load_vae(vae_name)
        base_latent = self._latent_spec(model, width, height)

        output_prefix = _safe_output_prefix(output_prefix)
        out_dir = os.path.join(folder_paths.get_output_directory(), output_prefix, "images")
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(os.path.dirname(out_dir), "manifest.json")
        # UI 预览要的是相对 output 目录的子路径，统一正斜杠
        subfolder = f"{output_prefix}/images"

        images = []
        manifest = []
        results = []
        t0 = time.time()

        for idx, (rel, ms, cs) in enumerate(entries):
            job_model, job_clip = model, clip

            if rel != "None":
                lora_path = folder_paths.get_full_path("loras", rel)
                if not lora_path:
                    print(f"[AnimaBatch] ❌ LoRA 未找到，跳过: {rel}")
                    continue
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                job_model, job_clip = comfy.sd.load_lora_for_models(
                    job_model, job_clip, lora_data,
                    ms if ms is not None else default_model_strength,
                    cs if cs is not None else default_clip_strength,
                )

            pos = _encode(job_clip, positive)
            neg = _encode(job_clip, negative)

            result = common_ksampler(
                model=job_model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler, scheduler=scheduler,
                positive=pos, negative=neg,
                latent={"samples": base_latent["samples"].clone()}, denoise=1.0,
            )
            latent_out = result[0]
            decoded = vae.decode(latent_out["samples"])
            image = self._to_image_bhwc(decoded)
            pil = self._decode_pil(image)

            base = "none" if rel == "None" else os.path.splitext(os.path.basename(rel))[0]
            fname = f"img_{idx:03d}_{base}.{image_format}"
            fpath = os.path.join(out_dir, fname)
            pil.save(fpath)
            images.append(image)
            results.append({"filename": fname, "subfolder": subfolder, "type": "output"})

            item = {
                "index": idx,
                "lora": rel,
                "model": unet_name,
                "clip": clip_name,
                "vae": vae_name,
                "clip_type": clip_type,
                "sampler": sampler,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "seed": seed,
                "filename": fname,
            }
            manifest.append(item)
            print(f"[AnimaBatch] ✅ [{idx + 1}/{len(entries)}] {rel} -> {fname}")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"session_name": output_prefix, "items": manifest}, f, ensure_ascii=False, indent=2)

        if not images:
            raise RuntimeError("没有生成任何图片：请检查 LoRA 文件名、模型路径和控制台错误信息。")
        print(f"[AnimaBatch] 完成 {len(images)} 张，用时 {time.time() - t0:.1f}s，目录: {out_dir}")
        # 必须是 {"ui": ..., "result": ...} 信封；直接返回 {"images": ...} 会被当成
        # 没有 'result' 的字典，导致 outputs 为空列表，缓存回填时 IndexError。
        return {"ui": {"images": results}, "result": (torch.cat(images, dim=0),)}


NODE_CLASS_MAPPINGS = {"AnimaLoRABatchTester": AnimaLoRABatchTester}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABatchTester": "Anima LoRA 批量测试"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
