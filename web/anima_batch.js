/**
 * Anima LoRA 批量测试：
 * 1. 按 lora_count 显隐 LoRA 下拉槽位；
 * 2. 正/负面提示词框可纵向拉伸，并把实测高度同步回控件布局；
 * 3. 不插入额外装饰 widget，避免工作流 widgets_values 按索引错位。
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "AnimaLoRABatchTester";
const SLOT_COUNT = 20;
const DEFAULT_COUNT = 10;
const PROMPT_WIDGETS = ["正面提示词 positive", "负面提示词 negative"];
const PROMPT_GAP_HEIGHT = 12;
const MIN_PROMPT_HEIGHT = 72;

function injectPromptStyle() {
  if (typeof document === "undefined") return;
  if (document.getElementById("anima-batch-prompt-resize")) return;

  const style = document.createElement("style");
  style.id = "anima-batch-prompt-resize";
  style.textContent = `
    textarea[data-anima-prompt] {
      resize: vertical !important;
      overflow: auto !important;
      min-height: ${MIN_PROMPT_HEIGHT}px;
      --comfy-widget-min-height: ${MIN_PROMPT_HEIGHT}px;
    }
  `;
  document.head.appendChild(style);
}

function syncPromptWidgetHeight(node, widget, el, extraBottom = 0) {
  const measured = Math.max(MIN_PROMPT_HEIGHT, Math.round(el.offsetHeight || MIN_PROMPT_HEIGHT));
  if (el.__animaLastHeight === measured && el.__animaLastGap === extraBottom) return;
  el.__animaLastHeight = measured;
  el.__animaLastGap = extraBottom;

  const margin = widget.margin ?? 10;
  // DOM 层按 computedHeight - 2*margin 给元素预留高度；
  // 正面框额外留出固定底部间距，避免与负面框贴在一起。
  const layoutHeight = measured + margin * 2 + extraBottom;

  el.style.marginBottom = `${extraBottom}px`;
  el.style.height = extraBottom > 0 ? `calc(100% - ${extraBottom}px)` : "100%";
  el.style.setProperty("--comfy-widget-min-height", `${measured}px`);
  el.style.setProperty("--comfy-widget-max-height", `${measured}px`);
  el.style.setProperty("--comfy-widget-height", `${measured}px`);

  const options = widget.options ?? (widget.options = {});
  options.getMinHeight = () => layoutHeight;
  options.getMaxHeight = () => layoutHeight;
  options.getHeight = () => layoutHeight;

  widget.computedHeight = layoutHeight;
  widget.computeSize = undefined;
  widget.computeLayoutSize = () => ({
    minHeight: layoutHeight,
    maxHeight: layoutHeight,
    minWidth: 0,
  });

  node.arrange?.();
  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

function setupPromptResizable(node) {
  injectPromptStyle();
  if (typeof ResizeObserver === "undefined") return;

  PROMPT_WIDGETS.forEach((name, index) => {
    const widget = node.widgets?.find((w) => w.name === name);
    const el = widget?.element;
    if (!el || !(el instanceof HTMLTextAreaElement)) return;

    el.dataset.animaPrompt = name;
    el.style.margin = "0";

    if (el.__animaPromptObserver) {
      el.__animaPromptObserver.disconnect();
    }

    const extraBottom = index === 0 ? PROMPT_GAP_HEIGHT : 0;
    const observer = new ResizeObserver(() => {
      try {
        syncPromptWidgetHeight(node, widget, el, extraBottom);
      } catch (error) {
        console.error("[AnimaBatch] 提示词框尺寸回写失败:", error);
      }
    });
    observer.observe(el);
    el.__animaPromptObserver = observer;

    syncPromptWidgetHeight(node, widget, el, extraBottom);
  });
}

const PARAMETER_DEFAULTS = {
  default_model_strength: 0.8,
  default_clip_strength: 1.0,
  include_none: true,
  sampler: "euler",
  scheduler: "simple",
  steps: 30,
  cfg: 4.0,
  width: 1024,
  height: 1536,
  output_prefix: "Anima-LoRA-batch",
  image_format: "webp",
  seed: 0,
};

function widgetIsValid(widget, name) {
  if (!widget) return false;
  const value = widget.value;
  if (["default_model_strength", "default_clip_strength"].includes(name)) {
    return Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 2;
  }
  if (name === "steps") return Number.isInteger(Number(value)) && Number(value) >= 1 && Number(value) <= 200;
  if (name === "cfg") return Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 20;
  if (["width", "height"].includes(name)) {
    return Number.isInteger(Number(value)) && Number(value) >= 64 && Number(value) <= 4096;
  }
  if (name === "seed") {
    return Number.isSafeInteger(Number(value)) && Number(value) >= 0;
  }
  if (name === "sampler" || name === "scheduler" || name === "image_format") {
    const values = widget.options?.values;
    return Array.isArray(values) && values.includes(value);
  }
  if (name === "output_prefix") return typeof value === "string" && value !== "randomize";
  if (name === "include_none") return typeof value === "boolean";
  return true;
}

function repairParameterWidgets(node) {
  const widgets = node.widgets ?? [];
  const byName = new Map(widgets.map((widget) => [widget.name, widget]));
  const names = Object.keys(PARAMETER_DEFAULTS);
  const missing = names.filter((name) => !byName.has(name));
  if (missing.length) {
    console.warn("[AnimaBatch] 参数控件缺失，无法执行自动修复:", missing);
    return;
  }

  const invalid = names.filter((name) => !widgetIsValid(byName.get(name), name));
  if (!invalid.length) return;

  console.warn("[AnimaBatch] 检测到工作流参数错位，已将参数组恢复为安全默认值。异常项:", invalid);
  for (const [name, value] of Object.entries(PARAMETER_DEFAULTS)) {
    const widget = byName.get(name);
    widget.value = value;
    if (widget.options) widget.options.value = value;
  }
  node.graph?.setDirtyCanvas(true, true);
  node.setDirtyCanvas?.(true, true);
}

function setWidgetHidden(widget, hidden) {
  if (!widget) return;
  const wanted = Boolean(hidden);
  const options = widget.options ?? (widget.options = {});
  if (Boolean(widget.hidden) === wanted && Boolean(options.hidden) === wanted) return;
  widget.hidden = wanted;
  options.hidden = wanted;
}

app.registerExtension({
  name: "AnimaBatch.GrowingLoRASlots",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;
    if (proto.__anima_lora_slots_wrapped) return;
    proto.__anima_lora_slots_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);

      try {
        const countWidget = this.widgets?.find((w) => w.name === "lora_count");
        const slotWidgets = Array.from({ length: SLOT_COUNT }, (_, i) =>
          this.widgets?.find((w) => w.name === `lora_${i + 1}`)
        ).filter(Boolean);

        const refold = () => {
          let count = Number(countWidget?.value ?? DEFAULT_COUNT);
          if (!Number.isFinite(count)) count = DEFAULT_COUNT;
          count = Math.min(Math.max(count, 1), SLOT_COUNT);
          slotWidgets.forEach((widget, index) => setWidgetHidden(widget, index + 1 > count));

          const computed = this.computeSize?.();
          if (computed) this.setSize([this.size[0], computed[1]]);
          this.graph?.setDirtyCanvas(true, true);
        };

        if (countWidget) {
          const originalCallback = countWidget.callback;
          countWidget.callback = function (...args) {
            const r = originalCallback?.apply(this, args);
            refold();
            return r;
          };
        }

        for (const widget of slotWidgets) {
          const originalCallback = widget.callback;
          widget.callback = function (...args) {
            const r = originalCallback?.apply(this, args);
            refold();
            return r;
          };
        }

        const originalConfigure = this.onConfigure;
        this.onConfigure = function (...args) {
          const r = originalConfigure?.apply(this, args);
          setupPromptResizable(this);
          repairParameterWidgets(this);
          refold();
          setTimeout(() => {
            setupPromptResizable(this);
            repairParameterWidgets(this);
            refold();
          }, 0);
          return r;
        };

        setupPromptResizable(this);
        repairParameterWidgets(this);
        refold();
      } catch (error) {
        console.error("[AnimaBatch] 槽位显隐初始化失败:", error);
      }

      return result;
    };
  },
});
