"""ImageHandler：图像媒体的 resolve / preprocess / to_text_view / to_content_part（spec 8.1）。

- resolve：引用 -> MediaAsset（魔数嗅探格式与 mime、读尺寸与字节数）
- preprocess：在 image_processor 预处理管线之上（校验+压缩+落盘产物，
  不覆盖原图，4.3），返回新 MediaAsset（path 指向预处理产物，meta 保留原始路径）
- to_text_view：级联方案的资产文本视图（B 的基础来源，D5 三要素：路径锚点+
  摘要+重看入口），完整 B 由 vision_extractor 编排渲染（4.6 格式）
- to_content_part：native 方案的图像段（data URL + detail，5.2）
- gif 取首帧由 image_processor._load_source 完成，resolved meta 中标注
  animated_first_frame（B 中标注"动图仅取首帧"，3.6）
"""

from __future__ import annotations

import base64
from pathlib import Path

from wdclaude.vision.image_processor import (
    DEFAULT_MAX_LONG_EDGE_PX,
    ImageDecodeError,
    image_info,
    preprocess_image,
    sniff_format,
)
from wdclaude.vision.media_types import (
    MediaAsset,
    MediaHandler,
    MediaKind,
)

DEFAULT_DETAIL = "auto"


class ImageHandler(MediaHandler):
    """图像 Handler（kind=MediaKind.IMAGE）。

    扩展名清单不在本类声明，统一由 config.yaml 的
    media.handlers.image.extensions 提供（单一事实来源，D10）。
    """

    kind = MediaKind.IMAGE

    def resolve(self, ref: str, *, source: str = "user_at") -> MediaAsset:
        """解析引用为 MediaAsset：魔数校验格式 + 读取尺寸/字节数。

        不存在 / 魔数不识别 / 打不开均抛 ImageDecodeError（上层按 4.9
        fail-fast 处理）。decode 失败信息即 fail-fast 面向用户信息的一部分。
        """
        path = Path(ref)
        if not path.is_absolute():
            raise ImageDecodeError(
                f"图像路径必须为绝对路径（与项目路径规范一致）: {ref}"
            )
        if not path.is_file():
            raise ImageDecodeError(f"图像文件不存在: {path}")
        sniff = sniff_format(path)
        if sniff is None:
            raise ImageDecodeError(
                f"图像魔数校验失败（扩展名与实际内容不符或格式不支持）: {path}"
            )
        fmt, mime = sniff
        dims, _ = image_info(path)
        return MediaAsset(
            path=path,
            kind=MediaKind.IMAGE,
            mime=mime,
            size_bytes=path.stat().st_size,
            dimensions=dims,
            source=source,
            meta={"original_format": fmt},
        )

    def preprocess(self, asset: MediaAsset) -> MediaAsset:
        """预处理（4.3）：校验+压缩，产物落盘；返回新 MediaAsset。

        预处理产物 path 指向 `.wdclaude/media/processed/<原名>_<hash>.png`，
        meta 中保留原始路径（asset.path 更新为产物路径，meta["original_path"]
        保存用户原图路径）。
        """
        original_path = asset.meta.get("original_path", str(asset.path))
        processed = preprocess_image(
            Path(original_path),
            max_long_edge_px=DEFAULT_MAX_LONG_EDGE_PX,
        )
        return MediaAsset(
            path=processed.path,
            kind=MediaKind.IMAGE,
            mime=processed.mime,
            size_bytes=processed.path.stat().st_size,
            dimensions=processed.dimensions,
            source=asset.source,
            meta={
                "original_path": str(original_path),
                "original_format": asset.meta.get("original_format", processed.fmt),
                "animated_first_frame": processed.animated_first_frame,
            },
        )

    def to_text_view(self, asset: MediaAsset, focus: str | None = None) -> str:
        """资产文本视图（级联方案 B 的基础来源）。

        完整 B 消息（分区标签、信任提示等）由 vision_extractor 按注入格式
        （4.6）编排；本方法只输出单图视图：路径锚点 + 基本信息 + 重看入口。
        focus 用于 view_image 特写场景的定向任务提示（extract_focus，
        4.4），常规 B 渲染无需传。
        """
        original = asset.meta.get("original_path", str(asset.path))
        animated = bool(asset.meta.get("animated_first_frame", False))
        lines = [
            f"[图] {original} ({asset.dimensions}, {asset.mime.split('/')[-1].upper()})",
        ]
        if animated:
            lines.append("[说明] 动图仅取首帧")
        if focus:
            lines.append(f"[焦点任务] {focus}")
        lines.append("[重看] 原图在磁盘，可用 view_image 重看（支持 region 裁剪特写）")
        return "\n".join(lines)

    def to_content_part(self, asset: MediaAsset, *, detail: str = DEFAULT_DETAIL) -> dict:
        """native 方案图像段（5.2 分段格式）：data URL + detail。"""
        img_path = asset.path
        try:
            payload = base64.b64encode(img_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise ImageDecodeError(f"读取图像失败: {img_path} ({exc})") from exc
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{asset.mime};base64,{payload}",
                "detail": detail,
            },
        }
