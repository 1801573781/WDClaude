"""远程图像 URL 获取（视觉链路的素材落地层）。

定位（与 URL 透传的分工）：
- 纯全图理解：URL 直接交给视觉模型（服务端下载），本模块不参与
- 需要本地像素：region 特写、裁剪、上采样、画框定位、批量预处理，
  必须先把远程图片落盘，再走既有像素管线

约束与防护：
- URL 长度上限 8192（与主流提供方一致）
- 流式下载 + 单图大小上限（默认 32 MiB），防止无限流拖死进程
- 内网/本地地址拒绝（SSRF 防护，与 web_fetch 同一条红线）。
  注意：此处只做字面量与 IP 段校验，DNS 重绑定不在防护范围内
- 魔数校验真实格式（不信任 Content-Type，也不信任扩展名）
- 同 URL 会话内复用缓存，文件名 = URL 哈希 + 真实扩展名
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

MAX_URL_LENGTH = 8192
MAX_IMAGE_BYTES = 32 * 1024 * 1024
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 60.0
CHUNK_SIZE = 64 * 1024
USER_AGENT = "WDClaude/2.0 (vision fetch)"

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# 后缀 -> MIME（真实格式判定优先，此表仅用于已确认格式后的标注）
SUFFIX_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


class RemoteFetchError(RuntimeError):
    """远程图像获取失败（消息可直接展示给主模型 / 用户）。"""


def is_remote_url(value: str) -> bool:
    """是否为 http(s) 远程地址（view_image / 引用识别共用判定）。"""
    return bool(_URL_RE.match((value or "").strip()))


def mime_for_path(path: Path | str) -> str:
    """按扩展名推断 MIME（本地图片兜底，未知后缀按 png 处理）。"""
    suffix = Path(path).suffix.lower().lstrip(".")
    return SUFFIX_MIME.get(suffix, "image/png")


def detect_image_format(head: bytes) -> tuple[str, str] | None:
    """按魔数嗅探真实图像格式，返回 (扩展名, MIME)；无法识别返回 None。

    格式由实际内容判断而非文件名或声明的 MIME（DeepSeek / GLM 同此语义）。
    """
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "gif", "image/gif"
    if head.startswith(b"BM"):
        return "bmp", "image/bmp"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None


def mime_for_bytes(payload: bytes) -> str:
    """字节流的 MIME（嗅探失败时回退 image/png）。"""
    sniffed = detect_image_format(payload[:32])
    return sniffed[1] if sniffed else "image/png"


def _assert_safe_url(url: str) -> None:
    """URL 白名单校验：长度、协议、非内网目标（SSRF 红线）。"""
    if len(url) > MAX_URL_LENGTH:
        raise RemoteFetchError(
            f"图像 URL 过长（{len(url)} > {MAX_URL_LENGTH} 字符），请改用 Base64 "
            f"或先把图片保存到本地"
        )
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise RemoteFetchError(f"仅支持 http/https 图像 URL: {url}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise RemoteFetchError(f"图像 URL 缺少主机名: {url}")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise RemoteFetchError(f"拒绝访问内网/本地地址: {host}")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return  # 域名形态，交由 requests 解析
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        raise RemoteFetchError(f"拒绝访问内网/本地地址: {host}")


def _cache_lookup(cache_dir: Path, digest: str) -> Path | None:
    """同 URL 缓存命中检查。"""
    for known in SUFFIX_MIME:
        candidate = cache_dir / f"{digest}.{known}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_image(url: str, cache_dir: Path) -> Path:
    """下载远程图像到本地缓存，返回落盘路径。

    失败一律抛 RemoteFetchError，消息含可读原因（下载失败属于输入问题，
    由调用方决定是返回工具文本还是终止会话）。
    """
    target_url = (url or "").strip()
    _assert_safe_url(target_url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(target_url.encode("utf-8")).hexdigest()[:16]

    cached = _cache_lookup(cache_dir, digest)
    if cached is not None:
        return cached

    tmp_path = cache_dir / f"{digest}.part"
    try:
        with requests.get(
            target_url,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        ) as resp:
            if resp.status_code != 200:
                raise RemoteFetchError(
                    f"图像 URL 下载失败（HTTP {resp.status_code}）: {target_url}"
                )
            body = bytearray()
            for chunk in resp.iter_content(CHUNK_SIZE):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > MAX_IMAGE_BYTES:
                    raise RemoteFetchError(
                        f"远程图像超过大小上限（{MAX_IMAGE_BYTES // (1024 * 1024)} MiB），"
                        f"请压缩后重试或改用本地文件: {target_url}"
                    )
    except RemoteFetchError:
        raise
    except requests.RequestException as exc:
        raise RemoteFetchError(f"图像 URL 下载失败: {target_url} ({exc})") from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if not body:
        raise RemoteFetchError(f"图像 URL 返回空内容: {target_url}")
    sniffed = detect_image_format(bytes(body[:32]))
    if sniffed is None:
        raise RemoteFetchError(
            f"远程内容不是可识别的图像（支持 PNG/JPEG/GIF/WebP/BMP）: {target_url}"
        )
    suffix, _mime = sniffed
    final_path = cache_dir / f"{digest}.{suffix}"
    final_path.write_bytes(bytes(body))
    return final_path


def materialize(source: str, cache_dir: Path) -> Path:
    """把 URL 或本地路径统一为本地图像路径（本地路径原样返回）。"""
    text = (source or "").strip()
    if is_remote_url(text):
        return download_image(text, cache_dir)
    return Path(text.replace("\\", "/"))
