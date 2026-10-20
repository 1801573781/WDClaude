"""提取协议编排、结果解析与 B 渲染（spec/multimodal_spec.md 4.1 / 4.4 / 4.5 / 4.6）。

职责：
- 编排视觉提取：预处理产物 -> vision_client 单轮调用（extract_full /
  extract_focus，prompt 位于 vision/prompts/）
- 解析 LLM 结构化输出为 4.5 数据结构（UncertainRegion /
  ImageExtraction / VisionExtractionResult）；解析失败降级为整段原文，
  不阻塞（4.5）
- 渲染 B 消息（4.6 完整格式：路径锚点、分区标签、不确定区域、意图画像、
  信任提示行 D7）
- 会话级视觉能力说明（4.6，仅首次多模态注入一次）
- 图像数超上限 fail-fast（4.9：超出 max_images_per_request 即终止）

错误处理：视觉调用失败由 VisionClient 抛 VisionFatalError 向上传播；
解码失败（ImageDecodeError）在本模块转换为 VisionFatalError（4.9）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from wdclaude.paths import paths
from wdclaude.vision.image_processor import ImageDecodeError
from wdclaude.vision.media_types import (
    MediaAsset,
    MediaKind,
    MediaRegistry,
)
from wdclaude.vision.remote_fetch import (
    RemoteFetchError,
    is_remote_url,
    materialize,
)
from wdclaude.vision.vision_client import VisionClient, VisionFatalError

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

DEFAULT_MAX_IMAGES = 5        # vision.max_images_per_request
DEFAULT_DETAIL = "auto"       # vision.detail

# 会话级视觉能力说明（4.6，首次多模态输入时在 B 之前注入一次）
SESSION_VISION_NOTICE = (
    "[系统提醒] 本会话已启用视觉能力。可用工具 view_image：参数 path(必填，"
    '绝对路径)、region(可选，归一化坐标 x0,y0,x1,y1 或语义描述如"红色按钮'
    "周围200px\")、focus(可选，定向任务)。用于重看原图或裁剪特写。"
)

# B 尾部信任提示行（D7 信任校准 + view_image 能力声明）
TRUST_HINT = (
    "[提示] 以上为多模态模型转述，OCR 可能有误，与代码实际内容矛盾时请以 "
    "file_view 为准。原图在磁盘，可用 view_image 重看（支持 region 裁剪特写）。"
)

PURE_IMAGE_HEADER = "[用户未提供文字说明，以下为图像内容提取与意图画像]"

# 纯图判定：A 为空或仅空白/标点（3.3）
_PUNCT_STRIP = " \t\r\n，。；、！？：,.;:!?~·…—-（）()【】[]{}\"'`"


# ----------------------------------------------------------------------
# 数据结构（4.5）
# ----------------------------------------------------------------------
@dataclass
class UncertainRegion:
    """无法可靠识别的区域（归一化 0-1 坐标 + 原因），两阶段协议触发信号。"""

    x0: float
    y0: float
    x1: float
    y1: float
    reason: str

    def to_line(self) -> str:
        return f"({self.x0},{self.y0})-({self.x1},{self.y1}): {self.reason}"


@dataclass
class ImageExtraction:
    """单图提取结果。path 为原图磁盘绝对路径（view_image 重看锚点）。"""

    path: str
    dimensions: str = ""
    format: str = ""
    text_content: str = ""
    layout: str = ""
    details: str = ""
    uncertain_regions: list[UncertainRegion] = field(default_factory=list)
    animated_first_frame: bool = False
    raw: str = ""  # 解析失败时的整段原文（降级，不阻塞）


@dataclass
class VisionExtractionResult:
    """多图容器（4.5）。"""

    extractions: list[ImageExtraction] = field(default_factory=list)
    diff: str = ""                     # 多图间差异，单图为空
    intent_profile: str | None = None  # 仅 pure_image 模式非空

    def brief_summary(self) -> str:
        """B 摘要（记忆召回查询增强用，3.1：A + B 摘要 + 引用角色清单）。"""
        parts: list[str] = []
        for ex in self.extractions:
            head = ex.text_content.strip().splitlines()[0][:60] if ex.text_content.strip() else ""
            parts.append(f"{Path(ex.path).name}: {head}" if head else Path(ex.path).name)
        if self.diff:
            parts.append(f"diff: {self.diff.strip()[:80]}")
        if self.intent_profile:
            parts.append(f"intent: {self.intent_profile.strip()[:60]}")
        return "; ".join(parts)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def is_pure_image(text: str) -> bool:
    """纯图判定：A 为空或仅空白/标点（3.3）。"""
    return not (text or "").strip(_PUNCT_STRIP)


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


_UNCERTAIN_LINE = re.compile(
    r"\((\d*\.?\d+)\s*[,，]\s*(\d*\.?\d+)\)\s*[-–]\s*\((\d*\.?\d+)\s*[,，]\s*(\d*\.?\d+)\)\s*[:：]\s*(.+)"
)

_PER_IMAGE_HEADER = re.compile(r"=====\s*图\s*(\d+)\s*=====")
_SECTION_HEADER = re.compile(r"^\[(文字内容|布局|细节|不确定区域|差异|意图画像)\]\s*$", re.MULTILINE)
_SECTION_NAMES = ("文字内容", "布局", "细节", "不确定区域", "差异", "意图画像")


def _split_sections(text: str) -> dict[str, str]:
    """按分区标签切分文本（[文字内容]...[布局]... 形式）。"""
    sections: dict[str, str] = {}
    matches = list(_SECTION_HEADER.finditer(text))
    if not matches:
        return sections
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.setdefault(name, text[start:end].strip())
    return sections


def _parse_uncertain(text: str) -> list[UncertainRegion]:
    regions: list[UncertainRegion] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line in ("无", "none"):
            continue
        m = _UNCERTAIN_LINE.search(line)
        if not m:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in m.groups()[:4])
        except ValueError:
            continue
        if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append(UncertainRegion(x0=x0, y0=y0, x1=x1, y1=y1, reason=m.group(5).strip()))
    return regions


# ----------------------------------------------------------------------
# 提取器
# ----------------------------------------------------------------------
class VisionExtractor:
    """提取协议编排 + 结果解析 + B 渲染（4.1 流水线的视觉环节）。"""

    def __init__(
        self,
        client: VisionClient,
        *,
        detail: str = DEFAULT_DETAIL,
        max_images_per_request: int = DEFAULT_MAX_IMAGES,
    ) -> None:
        self.client = client
        self.detail = detail
        self.max_images_per_request = max_images_per_request

    # ------------------------------------------------------------------
    # 全景提取（阶段一，7.1）
    # ------------------------------------------------------------------
    def extract_full(self, assets: list[MediaAsset], *, pure_image: bool) -> VisionExtractionResult:
        """对预处理产物执行全景扫描提取（4.4 extract_full 协议）。

        - 多图：一次调用全部图像（每图独立描述 + 图间差异，4.4）
        - pure_image：prompt 拼接 intent_profile 附加段（3.5）
        - 图像数超上限：VisionFatalError（4.9）
        """
        if not assets:
            raise VisionFatalError("视觉提取收到空图像清单")
        if len(assets) > self.max_images_per_request:
            raise VisionFatalError(
                f"图像数超出上限（{len(assets)} > {self.max_images_per_request}），"
                f"请分批提供图片"
            )
        prompt = _load_prompt("extract_full.txt")
        if pure_image:
            prompt = prompt + "\n\n" + _load_prompt("intent_profile.txt")

        if len(assets) == 1:
            messages = self.client.build_image_message(
                prompt, assets[0].path, detail=self.detail
            )
        else:
            messages = self.client.build_multi_image_message(
                prompt, [a.path for a in assets], detail=self.detail
            )
        raw = self.client.chat(messages)
        return self._parse_full_output(raw, assets, pure_image=pure_image)

    # ------------------------------------------------------------------
    # 定点特写提取（阶段二，view_image 使用，7.2）
    # ------------------------------------------------------------------
    def extract_focus(self, cropped_path: Path | str, focus: str) -> str:
        """对裁剪上采样产物执行定点特写提取（extract_focus，4.4）。

        返回结构化文本（分区格式），由 view_image 包装为工具结果。
        """
        return self.extract_focus_source(Path(cropped_path), focus)

    def extract_focus_source(self, source: Path | str, focus: str) -> str:
        """定点提取，source 可为本地图像路径或 http(s) 图像 URL。

        URL 是否可透传由 vision_client 依据 main_mlm.provider 配置的能力
        判定（未声明支持则抛 VisionFatalError，由调用方降级为本地落地）。
        """
        prompt = _load_prompt("extract_focus.txt")
        messages = self.client.build_image_message(
            prompt + f"\n\nfocus 定向任务：{focus or '无（全区域结构化描述）'}",
            source,
            detail=self.detail,
        )
        return self.client.chat(messages)

    # ------------------------------------------------------------------
    # 解析（解析失败降级为整段原文，不阻塞，4.5）
    # ------------------------------------------------------------------
    def _parse_full_output(
        self,
        raw: str,
        assets: list[MediaAsset],
        *,
        pure_image: bool,
    ) -> VisionExtractionResult:
        result = VisionExtractionResult()
        # 按图头切分
        headers = list(_PER_IMAGE_HEADER.finditer(raw))
        if headers:
            blocks: list[str] = []
            for i, h in enumerate(headers):
                start = h.end()
                end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
                blocks.append(raw[start:end])
            if len(blocks) != len(assets):
                # 图头数与图像数不符 -> 整体降级
                return self._degrade(raw, assets, pure_image=pure_image)
        else:
            if len(assets) != 1:
                return self._degrade(raw, assets, pure_image=pure_image)
            blocks = [raw]

        for asset, block in zip(assets, blocks):
            ex = self._parse_one_image(block, asset)
            if ex is None:
                return self._degrade(raw, assets, pure_image=pure_image)
            result.extractions.append(ex)

        tail = blocks[-1] if blocks else raw
        # 差异 / 意图画像分区可能在最后一图块内（全局追加区）
        last_sections = _split_sections(tail)
        result.diff = last_sections.get("差异", "")
        if pure_image:
            profile = last_sections.get("意图画像", "")
            result.intent_profile = profile if profile else None
        return result

    def _parse_one_image(self, block: str, asset: MediaAsset) -> ImageExtraction | None:
        sections = _split_sections(block)
        if not sections:  # 无任何分区标签 -> 该图降级
            return None
        original = str(asset.meta.get("original_path", str(asset.path)))
        ex = ImageExtraction(
            path=original,
            dimensions=asset.dimensions,
            format=str(asset.meta.get("original_format", "PNG")).upper(),
            text_content=sections.get("文字内容", ""),
            layout=sections.get("布局", ""),
            details=sections.get("细节", ""),
            uncertain_regions=_parse_uncertain(sections.get("不确定区域", "")),
            animated_first_frame=bool(asset.meta.get("animated_first_frame", False)),
        )
        return ex

    def _degrade(
        self, raw: str, assets: list[MediaAsset], *, pure_image: bool
    ) -> VisionExtractionResult:
        """解析失败降级：整段原文放入首图 text_content，不阻塞（4.5）。"""
        result = VisionExtractionResult()
        for i, asset in enumerate(assets):
            original = str(asset.meta.get("original_path", str(asset.path)))
            ex = ImageExtraction(
                path=original,
                dimensions=asset.dimensions,
                format=str(asset.meta.get("original_format", "PNG")).upper(),
                raw=raw if i == 0 else "(提取原文已附于图1)",
            )
            if i == 0:
                ex.text_content = raw  # 整段原文降级
            result.extractions.append(ex)
        return result

    # ------------------------------------------------------------------
    # B 渲染（4.6 完整格式）
    # ------------------------------------------------------------------
    def render_b(self, result: VisionExtractionResult, *, pure_image: bool) -> str:
        """渲染 B 消息文本（role=user，[视觉提取] 前缀由注入方添加或含于此）。"""
        n = len(result.extractions)
        model = self.client.route.resolved_model_name
        lines: list[str] = [f"[视觉提取] x{n} (vision={model})"]
        if pure_image:
            lines.append(PURE_IMAGE_HEADER)
        for i, ex in enumerate(result.extractions, 1):
            lines.append(f"[图{i}] {ex.path} ({ex.dimensions}, {ex.format})")
            if ex.animated_first_frame:
                lines.append("[说明] 动图仅取首帧")
            if ex.text_content:
                lines.append(f"[文字内容]\n{ex.text_content}")
            if ex.layout:
                lines.append(f"[布局]\n{ex.layout}")
            details = (ex.details or "").strip()
            if details and details != "无":
                lines.append(f"[细节]\n{details}")
            if ex.uncertain_regions:
                body = "\n".join(r.to_line() for r in ex.uncertain_regions)
                lines.append(f"[不确定区域]\n{body}")
        if result.diff:
            lines.append(f"[差异]\n{result.diff}")
        if pure_image and result.intent_profile:
            lines.append(f"[意图画像]\n{result.intent_profile}")
        lines.append(TRUST_HINT)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 流水线入口（L1 判定 media_input 之后的链路，4.1）
# ----------------------------------------------------------------------
def run_vision_pipeline(
    paths: list[str],
    registry: MediaRegistry,
    extractor: VisionExtractor,
    *,
    pure_image: bool,
) -> tuple[VisionExtractionResult, str]:
    """级联方案完整流水线：resolve -> preprocess -> extract_full -> render B。

    返回 (VisionExtractionResult, b_message_text)。任何失败均以
    VisionFatalError 抛出（含失败路径清单与建议，4.9 fail-fast）：
    - 路径数超上限 -> 提示分批提供
    - 扩展名不属于任何已启用 kind -> 提示在 media.handlers 中开启（8.1）
    - 解码失败（魔数/EXIF/PIL）-> 转为 VisionFatalError
    - 视觉模型调用失败 -> VisionClient 抛出
    """
    if len(paths) > extractor.max_images_per_request:
        raise VisionFatalError(
            f"图像数超出上限（{len(paths)} > {extractor.max_images_per_request}），"
            f"请分批提供图片"
        )
    handler = registry.handler_for(MediaKind.IMAGE)
    if handler is None:
        raise VisionFatalError(
            "image 媒体处理器未装配（config.yaml 的 media.handlers.image 未启用），"
            "无法处理媒体输入"
        )
    assets: list[MediaAsset] = []
    failed: list[str] = []
    last_err: Exception | None = None
    for p in paths:
        # 远程图像 URL：全景提取需要本地像素（后缀判定、解码、尺寸统计），
        # 先落地到 media/download 缓存再走既有预处理链路
        local_path = p
        if is_remote_url(p):
            try:
                local_path = str(materialize(p, paths.media_download_dir))
            except RemoteFetchError as exc:
                raise VisionFatalError(str(exc)) from exc
        ext = Path(local_path).suffix.lower()
        kind = registry.resolve_kind(ext)
        if kind is None:
            raise VisionFatalError(
                f"不支持的媒体类型: {p}（扩展名 {ext or '(无)'} 不属于任何已配置"
                "的媒体类型；若为预留类型，可在 config.yaml 的 media.handlers "
                "中开启对应配置）"
            )
        if not registry.is_kind_enabled(kind):
            raise VisionFatalError(
                f"{kind.value} 类型支持尚未启用，可在 config.yaml 的 "
                f"media.handlers 中开启（引用: {p}）"
            )
        try:
            asset = handler.resolve(local_path)
            # 远程来源：保留原 URL 作为锚点，便于 view_image 重看与日志追溯
            if is_remote_url(p):
                asset.meta["original_path"] = p
                asset.meta["remote_url"] = p
            asset = handler.preprocess(asset)
            assets.append(asset)
        except ImageDecodeError as exc:
            failed.append(str(p))
            last_err = exc
        except VisionFatalError:
            raise
        except Exception as exc:  # 未预期异常同样 fail-fast
            failed.append(str(p))
            last_err = exc
    if failed:
        raise VisionFatalError(
            f"图像预处理失败，失败路径清单: {failed}；最后一次错误: {last_err}。"
            f"建议检查图片文件是否损坏或更换图片后重试"
        )
    try:
        result = extractor.extract_full(assets, pure_image=pure_image)
    except VisionFatalError as exc:
        raise VisionFatalError(
            f"{exc}；建议检查网络连通性、稍后重试或减少图片数量"
        ) from exc
    b_message = extractor.render_b(result, pure_image=pure_image)
    return result, b_message
