"""view_image 工具（spec/multimodal_spec.md 6.2 / 6.3 / 7.2 / 7.3 / 7.4）。

XML 协议（与现有工具协议一致）：

    <view_image path="D:/proj/screenshot.png" region="0.62,0.71,0.85,0.80" focus="逐字OCR报错堆栈"/>

参数：
- path（必填）：原图绝对路径
- region（可选）：归一化坐标 x0,y0,x1,y1（0-1 闭区间）或语义描述
  （"红色按钮周围200px"，经视觉旁路定位换算）；缺省 = 全图重看
- focus（可选）：定向任务提示，映射到 extract_focus prompt

双模式路由（6.3，对主模型透明）：
- cascade（默认）：裁剪落盘 -> main_mlm 特写提取 -> 返回结构化文本
  {"role":"user","content":"[TOOL_RESULT] 特写提取 ..."}
- native：裁剪落盘 -> 裁剪图作为图像段附加到工具结果消息返回
  {"role":"user","content":[{"type":"text",...},{"type":"image_url",...}]}

两阶段协议（第 7 章）：
- 特写链路：parse_region -> crop_region(padding 0.05) -> upscale(zoom_min_edge)
  -> 落盘 .wdclaude/media/view/
- 连续特写 >=2 次（同一图）附加画框定位图（7.3 防迷路）
- 特写产物路径回注占位符（`screenshot.png#crop=...`，7.2）

熔断（7.4，会话级）：
- view_image_max_zoom（默认 5）：特写次数上限
- view_image_token_budget（默认 50000）：视觉旁路累计 token 预算
- 达到上限返回警告文本，不再调用视觉服务
- view_image 计入现有工具轮次统计（last_tool_sig 熔断由 query_loop 天然覆盖）

错误处理：
- 参数错误（缺 path / region 非法）返回错误文本（主模型可修正后重试）
- 视觉模型调用失败抛 VisionFatalError -> 上层按 4.9 fail-fast 终止会话
"""

from __future__ import annotations

from pathlib import Path

from wdclaude.vision.image_processor import (
    RegionBox,
    RegionParseError,
    annotate_locator,
    crop_and_zoom,
    is_coordinate_region,
    parse_region,
    preprocess_image,
    resolve_semantic_region,
)
from wdclaude.vision.lifecycle import build_native_tool_result
from wdclaude.vision.media_types import MediaKind, MediaRegistry
from wdclaude.vision.vision_client import VisionClient, VisionFatalError
from wdclaude.vision.vision_extractor import VisionExtractor

TOOL_NAME = "view_image"

DEFAULT_MAX_ZOOM = 5          # vision.view_image_max_zoom（会话级）
DEFAULT_TOKEN_BUDGET = 50000  # vision.view_image_token_budget（会话级）
BUDGET_EXHAUSTED_TEXT = "[视觉预算耗尽] 请基于已提取信息决策"


class ViewImageTool:
    """view_image 工具执行器（每会话一个实例，持有会话级熔断状态）。

    两方案差异（cascade / native）封装在本工具内部路由，主模型无需
    感知所在方案（6.3）。
    """

    def __init__(
        self,
        registry: MediaRegistry,
        extractor: VisionExtractor,
        *,
        mode: str = "cascade",
        max_zoom: int = DEFAULT_MAX_ZOOM,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.registry = registry
        self.extractor = extractor
        self.mode = "native" if mode == "native" else "cascade"
        self.max_zoom = max_zoom
        self.token_budget = token_budget
        # 会话级熔断状态（7.4）
        self.zoom_count = 0
        self.token_used = 0
        self._last_path: str | None = None
        self._consecutive_same_path = 0
        self._install_usage_hook()

    # ------------------------------------------------------------------
    # token 预算累计（包装 client.on_usage，7.4）
    # ------------------------------------------------------------------
    def _install_usage_hook(self) -> None:
        client = self.extractor.client
        original = client.on_usage

        def hook(category: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
            self.token_used += max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
            if original is not None:
                try:
                    original(category, model, prompt_tokens, completion_tokens)
                except Exception:
                    pass

        client.on_usage = hook

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def execute(self, params: dict) -> dict:
        """执行 view_image，返回工具结果消息 dict（role=user）。

        - 参数错误：返回 [TOOL_RESULT] 错误说明（不终止，主模型可修正重试）
        - 熔断：返回预算耗尽警告文本（7.4）
        - 视觉调用失败：VisionFatalError 向上传播（4.9 fail-fast）
        """
        path = str(params.get("path", "") or "").strip()
        if not path:
            return self._text_result("[view_image] 缺少必填参数 path（原图绝对路径）")
        normalized = path.replace("\\", "/")
        src = Path(normalized)
        if not src.is_file():
            return self._text_result(f"[view_image] 图像文件不存在: {normalized}")

        # 熔断检查（7.4）：次数或 token 预算任一达上限即拒绝
        if self.zoom_count >= self.max_zoom or self.token_used >= self.token_budget:
            return self._text_result(BUDGET_EXHAUSTED_TEXT)

        region_raw = str(params.get("region", "") or "").strip()
        focus = str(params.get("focus", "") or "").strip()

        # region 解析（6.2）：归一化坐标优先，语义描述经视觉旁路换算
        box: RegionBox | None = None
        region_label = "全图"
        if region_raw:
            if is_coordinate_region(region_raw):
                try:
                    box = parse_region(region_raw)
                except RegionParseError as exc:
                    return self._text_result(f"[view_image] {exc}")
            else:
                try:
                    box, region_label = self._resolve_semantic(normalized, region_raw)
                except RegionParseError as exc:
                    return self._text_result(f"[view_image] 语义 region 定位失败: {exc}")

        # 特写计数与连续同图判定（7.3 画框触发条件）
        if normalized == self._last_path:
            self._consecutive_same_path += 1
        else:
            self._consecutive_same_path = 1
            self._last_path = normalized
        self.zoom_count += 1

        try:
            if box is None:
                # 全图重看：走预处理管线控制分辨率（分辨率第一杠杆，5.5）
                processed = preprocess_image(src)
                target_path: Path = processed.path
                crop_tag = "全图"
            else:
                # 特写链路（7.2）：padding -> 上采样 -> 落盘 view 目录
                target_path = crop_and_zoom(src, box)
                crop_tag = f"{src.name}#crop={box.x0},{box.y0},{box.x1},{box.y1}"
        except RegionParseError as exc:
            return self._text_result(f"[view_image] {exc}")
        except Exception as exc:
            raise VisionFatalError(
                f"view_image 图像处理失败: {normalized} ({exc})；"
                f"建议检查图片文件是否损坏或更换图片后重试"
            ) from exc

        # 画框定位图（7.3）：连续特写 >=2 次附加
        locator_path: Path | None = None
        if box is not None and self._consecutive_same_path >= 2:
            try:
                locator_path = annotate_locator(src, box)
            except Exception:
                locator_path = None  # 定位图失败不阻塞主流程

        # 双模式路由（6.3）
        if self.mode == "native":
            return self._run_native(normalized, target_path, crop_tag, box, locator_path, focus)
        return self._run_cascade(normalized, target_path, crop_tag, box, locator_path, focus)

    # ------------------------------------------------------------------
    # 语义 region 定位（6.2：经视觉旁路换算归一化坐标）
    # ------------------------------------------------------------------
    def _resolve_semantic(self, path: str, description: str) -> tuple[RegionBox, str]:
        client: VisionClient = self.extractor.client

        def vision_call(prompt: str, image_path: Path) -> str:
            messages = VisionClient.image_message(prompt, image_path)
            return client.chat(messages)

        return resolve_semantic_region(Path(path), description, vision_call)

    # ------------------------------------------------------------------
    # cascade 模式（6.3）：main_mlm 特写提取 -> 结构化文本
    # ------------------------------------------------------------------
    def _run_cascade(
        self,
        original: str,
        target: Path,
        crop_tag: str,
        box: RegionBox | None,
        locator: Path | None,
        focus: str,
    ) -> dict:
        parts = [f"[TOOL_RESULT] 特写提取 {crop_tag}"]
        try:
            extracted = self.extractor.extract_focus(target, focus or "无（全区域结构化描述）")
        except VisionFatalError as exc:
            raise VisionFatalError(f"{exc}；建议检查网络连通性或稍后重试") from exc
        parts.append(extracted.strip())
        if box is not None:
            anchor = f"[位置锚] 原图 {box.to_anchor_str()} 区域（{crop_tag.split('#')[0]}）"
            parts.append(anchor)
        if locator is not None:
            parts.append(f"[画框定位] {locator}（红框为当前特写区域在原图中的位置）")
        return {"role": "user", "content": "\n".join(parts)}

    # ------------------------------------------------------------------
    # native 模式（6.3）：裁剪图作为图像段附加
    # ------------------------------------------------------------------
    def _run_native(
        self,
        original: str,
        target: Path,
        crop_tag: str,
        box: RegionBox | None,
        locator: Path | None,
        focus: str,
    ) -> dict:
        text_parts = [f"[TOOL_RESULT] 特写图已注入 {crop_tag}"]
        if focus:
            text_parts.append(f"[焦点任务] {focus}")
        if box is not None:
            text_parts.append(f"[位置锚] 原图 {box.to_anchor_str()} 区域")
        images = [target]
        if locator is not None:
            text_parts.append(f"[画框定位] {locator}（红框为当前特写区域在原图中的位置）")
            images.append(locator)
        return build_native_tool_result("\n".join(text_parts), images)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _text_result(text: str) -> dict:
        return {"role": "user", "content": text}

    # ------------------------------------------------------------------
    # 状态查询（诊断/日志用）
    # ------------------------------------------------------------------
    def budget_status(self) -> str:
        return (
            f"view_image: zoom={self.zoom_count}/{self.max_zoom}, "
            f"tokens={self.token_used}/{self.token_budget}"
        )
