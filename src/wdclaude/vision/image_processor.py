"""图像预处理与几何工具（spec/multimodal_spec.md 4.3 / 7.2 / 7.3）。

职责：
- 预处理管线：魔数校验、EXIF 方向修正、最长边压缩（LANCZOS）、产物落盘
  `.wdclaude/media/processed/<原名>_<hash>.png`（不覆盖原图，4.3）
- region 解析：归一化坐标 `x0,y0,x1,y1`（0-1）与语义描述（经视觉旁路定位，
  6.2 / 7.2）
- 裁剪链路：crop_region（padding_ratio=0.05 边缘外扩）→ upscale_to_min_edge
  （最短边上采样至 zoom_min_edge，默认 1024）→ 落盘
  `.wdclaude/media/view/<原名>_r<N>.png`（7.2）
- 画框定位图：原图缩至长边 512 为底图、ROI 3px 红框合成落盘（7.3）

解码类失败统一抛 ImageDecodeError，由上层按 4.9 fail-fast 转为
VisionFatalError 终止会话（本模块不直接终止）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageOps

# ----------------------------------------------------------------------
# 常量（与 spec 10.1 配置项一一对应，此处仅为缺省值）
# ----------------------------------------------------------------------
DEFAULT_MAX_LONG_EDGE_PX = 1500     # vision.max_long_edge_px
DEFAULT_ZOOM_MIN_EDGE = 1024        # vision.zoom_min_edge
DEFAULT_PADDING_RATIO = 0.05        # crop 边缘外扩比例（7.2）
LOCATOR_BASE_LONG_EDGE = 512        # 画框定位图底图长边（7.3）
LOCATOR_BORDER_PX = 3               # 画框线宽（7.3）

# 魔数表：格式名 -> (magic 头, mime)；防伪扩展名（4.3 第 1 条）
MAGIC_SIGNATURES: list[tuple[str, bytes, str]] = [
    ("PNG", b"\x89PNG\r\n\x1a\n", "image/png"),
    ("JPEG", b"\xff\xd8\xff", "image/jpeg"),
    ("GIF", b"GIF87a", "image/gif"),
    ("GIF", b"GIF89a", "image/gif"),
    ("BMP", b"BM", "image/bmp"),
    ("WEBP", b"RIFF", "image/webp"),  # 需再校验偏移 8 处的 WEBP 标识
]

# 语义 region 定位 prompt（region 为语义描述时经视觉旁路换算为归一化坐标，6.2）
LOCATOR_PROMPT = """你是图像区域定位器。给定一张图像与一段区域描述，请在图中找到该区域，\
并输出覆盖该区域（含描述中提到的外扩范围，如"周围200px"）的归一化外接框。

要求：
- 坐标为归一化值，取值范围 [0,1]，原点为图像左上角，x 向右、y 向下
- box 必须完整包住描述目标；若描述含"周围Npx"类外扩，请按原图像素尺寸换算后并入 box
- 只输出 JSON，不要输出任何其他文字，格式：
{"label": "<区域的一句话命名>", "box": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}}

区域描述：{description}"""


class ImageDecodeError(RuntimeError):
    """图像解码失败（魔数校验 / EXIF / PIL open 失败），按 4.9 转为会话终止。"""


class RegionParseError(ValueError):
    """region 参数解析失败（坐标非法 / 语义定位输出无法解析）。"""


# ----------------------------------------------------------------------
# 路径工具（.wdclaude/media 目录体系）
# ----------------------------------------------------------------------
def find_project_root() -> Path:
    """定位项目根目录：环境变量优先，其次从包位置向上探测，最后回退 cwd。"""
    env = os.environ.get("WDCLAUDE_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / ".wdclaude").is_dir() or (cand / "pyproject.toml").is_file():
            return cand
    return Path.cwd().resolve()


def media_root() -> Path:
    """媒体工作根目录 `<project_root>/.wdclaude/media`（图像永不离盘原则的落盘区）。"""
    root = find_project_root() / ".wdclaude" / "media"
    root.mkdir(parents=True, exist_ok=True)
    return root


def processed_dir() -> Path:
    """预处理产物目录（4.3 第 3 条）。"""
    d = media_root() / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def view_dir() -> Path:
    """特写/画框产物目录（7.2 / 7.3）。"""
    d = media_root() / "view"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RegionBox:
    """归一化外接框，四值均在 [0,1] 闭区间，原点左上。"""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_str(self) -> str:
        return f"{self.x0:.4g},{self.y0:.4g},{self.x1:.4g},{self.y1:.4g}".rstrip("0").rstrip(".") if False else \
            f"{self.x0},{self.y0},{self.x1},{self.y1}"

    def to_anchor_str(self) -> str:
        """锚点形式 `(x0,y0)-(x1,y1)`（4.6 / 7.3 文本定位锚）。"""
        return f"({self.x0:.4g},{self.y0:.4g})-({self.x1:.4g},{self.y1:.4g})"


@dataclass
class ProcessedImage:
    """预处理产物描述（4.3）：path 指向产物，original_path 保留原始路径。"""

    path: Path
    original_path: Path
    dimensions: str          # "1920x1080"
    fmt: str                 # "PNG"
    animated_first_frame: bool = False  # gif 取首帧时为 True（B 中标注"动图仅取首帧"）

    @property
    def mime(self) -> str:
        return "image/png"


# ----------------------------------------------------------------------
# 魔数与基础信息
# ----------------------------------------------------------------------
def sniff_format(path: Path) -> tuple[str, str] | None:
    """魔数嗅探，返回 (格式名, mime)；无法识别返回 None（防伪扩展名）。"""
    try:
        head = path.read_bytes()[:16]
    except OSError as exc:
        raise ImageDecodeError(f"读取图像文件失败: {path} ({exc})") from exc
    for fmt, magic, mime in MAGIC_SIGNATURES:
        if head.startswith(magic):
            if fmt == "WEBP" and head[8:12] != b"WEBP":
                continue
            return fmt, mime
    return None


def image_info(path: Path) -> tuple[str, str]:
    """读取图像尺寸与格式名，返回 ("1920x1080", "PNG")。解码失败抛 ImageDecodeError。"""
    try:
        with Image.open(path) as img:
            w, h = img.size
            fmt = (img.format or "UNKNOWN").upper()
    except Exception as exc:
        raise ImageDecodeError(f"图像打开失败: {path} ({exc})") from exc
    return f"{w}x{h}", fmt


# ----------------------------------------------------------------------
# 预处理管线（4.3）
# ----------------------------------------------------------------------
def _load_source(path: Path) -> tuple[Image.Image, bool]:
    """加载原图：魔数校验 + EXIF 方向修正 + gif 取首帧。"""
    sniff = sniff_format(path)
    if sniff is None:
        raise ImageDecodeError(
            f"图像魔数校验失败（扩展名与实际内容不符或格式不支持）: {path}"
        )
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise ImageDecodeError(f"图像解码失败: {path} ({exc})") from exc
    animated_first = False
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)  # 动图仅取首帧（3.6）
            animated_first = True
    except Exception:
        pass
    try:
        img = ImageOps.exif_transpose(img)  # EXIF 方向修正
    except Exception as exc:
        raise ImageDecodeError(f"EXIF 方向修正失败: {path} ({exc})") from exc
    if img.mode == "P":
        img = img.convert("RGBA")
    return img, animated_first


def preprocess_image(
    src: Path,
    *,
    max_long_edge_px: int = DEFAULT_MAX_LONG_EDGE_PX,
) -> ProcessedImage:
    """完整预处理（4.3）：魔数/EXIF/首帧 → 最长边压缩 → 产物落盘 PNG。

    产物路径 `.wdclaude/media/processed/<原名>_<hash8>.png`，hash 取原文件
    内容 sha1 前 8 位（同图幂等、不覆盖原图）。
    """
    src = Path(src)
    if not src.is_file():
        raise ImageDecodeError(f"图像文件不存在: {src}")
    img, animated_first = _load_source(src)
    w, h = img.size
    if max(w, h) > max_long_edge_px:
        scale = max_long_edge_px / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        w, h = img.size
    try:
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:8]
    except OSError as exc:
        raise ImageDecodeError(f"读取图像文件失败: {src} ({exc})") from exc
    out = processed_dir() / f"{src.stem}_{digest}.png"
    try:
        img.save(out, format="PNG")
    except Exception as exc:
        raise ImageDecodeError(f"预处理产物写入失败: {out} ({exc})") from exc
    return ProcessedImage(
        path=out,
        original_path=src,
        dimensions=f"{w}x{h}",
        fmt="PNG",
        animated_first_frame=animated_first,
    )


# ----------------------------------------------------------------------
# region 解析（6.2 / 7.2）
# ----------------------------------------------------------------------
_REGION_RE = re.compile(r"^\s*(\d*\.?\d+)\s*[,，]\s*(\d*\.?\d+)\s*[,，]\s*(\d*\.?\d+)\s*[,，]\s*(\d*\.?\d+)\s*$")


def parse_region(text: str) -> RegionBox:
    """解析归一化坐标 region `x0,y0,x1,y1`（兼容中英文逗号与空格）。

    非法输入抛 RegionParseError；语义描述请先经 resolve_semantic_region。
    """
    m = _REGION_RE.match(text or "")
    if not m:
        raise RegionParseError(
            f"region 坐标格式错误: {text!r}，应为归一化坐标 x0,y0,x1,y1（0-1）"
        )
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    for name, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
        if not 0.0 <= v <= 1.0:
            raise RegionParseError(f"region 坐标 {name}={v} 超出 [0,1] 闭区间")
    if x1 <= x0 or y1 <= y0:
        raise RegionParseError(f"region 坐标非法（x1<=x0 或 y1<=y0）: {text!r}")
    return RegionBox(x0=x0, y0=y0, x1=x1, y1=y1)


def is_coordinate_region(text: str) -> bool:
    """判断 region 文本是否为归一化坐标形式（非坐标即语义描述）。"""
    try:
        parse_region(text)
        return True
    except RegionParseError:
        return False


def _extract_json(text: str) -> dict:
    """从模型输出中提取首个 JSON 对象（容忍 ```json 围栏与前后杂文）。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RegionParseError(f"无法从输出中解析 JSON: {text[:200]!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RegionParseError(f"JSON 解析失败: {exc}") from exc


def resolve_semantic_region(
    image_path: Path,
    description: str,
    vision_call: Callable[[str, Path], str],
) -> tuple[RegionBox, str]:
    """语义 region 解析（6.2）：经视觉旁路定位换算为归一化坐标。

    vision_call(prompt, image_path) -> 模型输出文本（由调用方注入旁路客户端），
    返回 (RegionBox, 区域一句话命名)。
    """
    prompt = LOCATOR_PROMPT.replace("{description}", description.strip())
    raw = vision_call(prompt, Path(image_path))
    data = _extract_json(raw)
    try:
        box_raw = data["box"]
        box = RegionBox(
            x0=float(box_raw["x0"]),
            y0=float(box_raw["y0"]),
            x1=float(box_raw["x1"]),
            y1=float(box_raw["y1"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RegionParseError(f"语义定位输出缺少合法 box 字段: {exc}") from exc
    box = RegionBox(
        x0=min(max(box.x0, 0.0), 1.0),
        y0=min(max(box.y0, 0.0), 1.0),
        x1=min(max(box.x1, 0.0), 1.0),
        y1=min(max(box.y1, 0.0), 1.0),
    )
    if box.x1 <= box.x0 or box.y1 <= box.y0:
        raise RegionParseError(f"语义定位 box 非法（x1<=x0 或 y1<=y0）: {data}")
    label = str(data.get("label", description.strip()))
    return box, label


# ----------------------------------------------------------------------
# 裁剪 / 上采样 / 落盘（7.2）
# ----------------------------------------------------------------------
def crop_region(
    img: Image.Image,
    box: RegionBox,
    *,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> Image.Image:
    """按归一化框裁剪，含 padding_ratio 边缘外扩（避免裁掉表头/上下文）。"""
    w, h = img.size
    left = box.x0 * w
    top = box.y0 * h
    right = box.x1 * w
    bottom = box.y1 * h
    pad = padding_ratio * max(right - left, bottom - top)
    left = max(0.0, left - pad)
    top = max(0.0, top - pad)
    right = min(float(w), right + pad)
    bottom = min(float(h), bottom + pad)
    if right - left < 1 or bottom - top < 1:
        raise RegionParseError(f"裁剪区域过小: {box}")
    return img.crop((round(left), round(top), round(right), round(bottom)))


def upscale_to_min_edge(img: Image.Image, min_edge: int = DEFAULT_ZOOM_MIN_EDGE) -> Image.Image:
    """最短边上采样至 min_edge（摊满 tile 预算）；已达标则原样返回。"""
    w, h = img.size
    if min(w, h) >= min_edge:
        return img
    scale = min_edge / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def crop_and_zoom(
    src: Path,
    box: RegionBox,
    *,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    min_edge: int = DEFAULT_ZOOM_MIN_EDGE,
) -> Path:
    """裁剪链路（7.2）：解析后的 box → 外扩裁剪 → 上采样 → 落盘 view 目录。

    产物名 `<原名>_r<hash6>.png`，hash 取 box 四值（同区域幂等）。
    """
    img, _ = _load_source(Path(src))
    cropped = crop_region(img, box, padding_ratio=padding_ratio)
    zoomed = upscale_to_min_edge(cropped, min_edge)
    tag = hashlib.sha1(box.to_str().encode("utf-8")).hexdigest()[:6]
    out = view_dir() / f"{Path(src).stem}_r{tag}.png"
    try:
        zoomed.save(out, format="PNG")
    except Exception as exc:
        raise ImageDecodeError(f"特写产物写入失败: {out} ({exc})") from exc
    return out


# ----------------------------------------------------------------------
# 画框定位图（7.3，防"放大镜迷路"）
# ----------------------------------------------------------------------
def annotate_locator(src: Path, box: RegionBox) -> Path:
    """生成画框定位图：原图缩至长边 512 为底图、ROI 3px 红框、合成落盘。"""
    img, _ = _load_source(Path(src))
    w, h = img.size
    scale = LOCATOR_BASE_LONG_EDGE / max(w, h)
    base = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    base = base.convert("RGB")
    bw, bh = base.size
    sx, sy = bw / w, bh / h
    left = max(0, round(box.x0 * w * sx))
    top = max(0, round(box.y0 * h * sy))
    right = min(bw, round(box.x1 * w * sx))
    bottom = min(bh, round(box.y1 * h * sy))
    # 边界内缩 1px，保证 3px 边框可见
    left = min(left, max(0, bw - LOCATOR_BORDER_PX - 1))
    top = min(top, max(0, bh - LOCATOR_BORDER_PX - 1))
    right = max(right, min(bw, left + LOCATOR_BORDER_PX + 1))
    bottom = max(bottom, min(bh, top + LOCATOR_BORDER_PX + 1))
    draw = ImageDraw.Draw(base)
    draw.rectangle(
        (left, top, right, bottom),
        outline=(255, 0, 0),
        width=LOCATOR_BORDER_PX,
    )
    tag = hashlib.sha1(box.to_str().encode("utf-8")).hexdigest()[:6]
    out = view_dir() / f"{Path(src).stem}_loc{tag}.png"
    try:
        base.save(out, format="PNG")
    except Exception as exc:
        raise ImageDecodeError(f"画框定位图写入失败: {out} ({exc})") from exc
    return out
