"""视觉模型旁路客户端（spec/multimodal_spec.md 4.2，决策 D6）。

参照 embedding_client 的旁路模式：独立类、单轮非流式（stream=False）、
OpenAI 兼容接口，不进入主 query loop 的流式通道。

路由配置（两级读取，10.2，与 main_model / simple_chat 同机制）：
1. 读 model_key.yaml 的 `main_mlm.provider`（如 "GLM-main-mlm"）
2. 以该值为节名读取服务商连接参数：api_key / base_url / model_name /
   extra_body；main_mlm 节自身可携带 extra_body 覆盖

图像以 data URL（base64）随消息传入：
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

错误处理（D8 / 4.9）：
- 重试 2 次；最终失败抛出 VisionFatalError，由上层执行会话终止
- token 消耗计入 token_statistics 的独立分类（vision，11 章）

依赖注入：配置解析与客户端实例化由调用方完成（config_loader 产出
SimpleNamespace；OpenAI 客户端在首次调用时惰性创建）。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from wdclaude.vision.media_types import cfg_get
from wdclaude.vision.remote_fetch import is_remote_url, mime_for_path

DEFAULT_RETRIES = 2       # 重试 2 次（4.2）
DEFAULT_TIMEOUT = 120.0   # 单次调用超时（秒）
TOKEN_CATEGORY = "vision"  # token_statistics 独立分类（11 章）

# 提供方家族识别（决定外部 URL 能力与 base64 形式的保守默认值）。
# 具体模型由 model_key.yaml 的 main_mlm.provider 配置，此处仅按名称特征
# 归类，未识别的提供方一律按 generic 处理（保守：不透传 URL）。
_FAMILY_HINTS = (
    (("deepseek",), "deepseek"),
    (("glm", "zhipu", "bigmodel", "z.ai"), "glm"),
)

# 家族默认能力: family -> (supports_external_url, base64_mode)
#   supports_external_url：是否允许把 http(s) URL 直接交给服务端抓取
#   base64_mode："data_url" 为 data:image/...;base64,... ；"raw" 为裸 base64
_FAMILY_DEFAULTS = {
    "deepseek": (True, "data_url"),
    "glm": (True, "raw"),
    "generic": (False, "data_url"),
}


def detect_family(*names: str) -> str:
    """从提供方名 / 模型名推断家族（未识别返回 generic）。"""
    blob = " ".join(str(n or "").lower() for n in names)
    for needles, family in _FAMILY_HINTS:
        if any(n in blob for n in needles):
            return family
    return "generic"


class VisionFatalError(RuntimeError):
    """视觉链路致命错误（D8 fail-fast）→ 终止会话。

    错误信息含失败路径清单与最后一次 API 报错摘要（4.9）。
    """


@dataclass
class MlmRoute:
    """main_mlm 两级路由解析结果（10.2）。"""

    provider: str
    api_key: str
    base_url: str
    model_name: str
    extra_body: dict | None = None
    family: str = "generic"
    supports_external_url: bool = False
    base64_mode: str = "data_url"

    @property
    def resolved_model_name(self) -> str:
        return self.model_name

    @property
    def prefers_raw_base64(self) -> bool:
        return self.base64_mode == "raw"


def resolve_mlm_route(model_key_cfg: Any) -> MlmRoute:
    """解析 main_mlm 两级路由（model_key.yaml 结构）。

    model_key_cfg：model_key 配置对象（SimpleNamespace 或 dict），含
    `main_mlm.provider` 与服务商节（如 GLM-main-mlm）。
    """
    mlm = cfg_get(model_key_cfg, "main_mlm")
    provider = str(cfg_get(mlm, "provider", "") or "").strip()
    if not provider:
        raise VisionFatalError(
            "model_key.yaml 缺少 main_mlm.provider 配置（多模态模型路由），"
            "请按 spec/multimodal_spec.md 10.2 补充 main_mlm 节"
        )
    section = cfg_get(model_key_cfg, provider)
    api_key = str(cfg_get(section, "api_key", "") or "")
    base_url = str(cfg_get(section, "base_url", "") or "")
    model_name = str(cfg_get(section, "model_name", "") or "")
    if not api_key or not base_url or not model_name:
        raise VisionFatalError(
            f"model_key.yaml 服务商节 {provider!r} 配置不完整"
            f"（api_key/base_url/model_name），无法建立视觉旁路客户端"
        )
    # main_mlm 节自身可携带 extra_body 覆盖（机制同 simple_chat，10.2）
    extra_body = cfg_get(section, "extra_body", None)
    mlm_extra = cfg_get(mlm, "extra_body", None)
    if mlm_extra:
        merged = dict(extra_body) if isinstance(extra_body, dict) else {}
        merged.update(mlm_extra if isinstance(mlm_extra, dict) else {})
        extra_body = merged
    # 家族默认能力 + main_mlm 节的可选覆盖（未知提供方保守处理）
    family = detect_family(provider, model_name)
    default_url, default_mode = _FAMILY_DEFAULTS.get(family, _FAMILY_DEFAULTS["generic"])
    override_url = cfg_get(mlm, "supports_external_url", None)
    override_mode = cfg_get(mlm, "base64_mode", None)
    supports_external_url = (
        bool(override_url) if isinstance(override_url, bool) else default_url
    )
    base64_mode = str(override_mode).strip().lower() if override_mode else default_mode
    if base64_mode not in ("data_url", "raw"):
        base64_mode = default_mode

    return MlmRoute(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        extra_body=extra_body if isinstance(extra_body, dict) else None,
        family=family,
        supports_external_url=supports_external_url,
        base64_mode=base64_mode,
    )


class VisionClient:
    """视觉模型旁路客户端（单轮非流式，D6）。"""

    def __init__(self, route: MlmRoute, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.route = route
        self.timeout = timeout
        self._client: OpenAI | None = None
        # token 统计回调：签名 (category: str, model_name: str,
        # prompt_tokens: int, completion_tokens: int) -> None，由调用方
        # 注入 token_statistics 的记录函数（11 章，独立 vision 分类）
        self.on_usage: Callable[[str, str, int, int], None] | None = None

    # ------------------------------------------------------------------
    # 客户端管理
    #----------------------------------------------------------------
    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            try:
                self._client = OpenAI(
                    api_key=self.route.api_key,
                    base_url=self.route.base_url,
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise VisionFatalError(
                    f"视觉旁路客户端初始化失败 (provider={self.route.provider}): {exc}"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def _report_usage(self, usage: Any) -> None:
        if self.on_usage is None or usage is None:
            return
        try:
            self.on_usage(
                TOKEN_CATEGORY,
                self.route.model_name,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        except Exception:
            pass  # 统计失败不阻塞视觉链路

    def chat(
        self,
        messages: list[dict],
        *,
        retries: int = DEFAULT_RETRIES,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        """单轮非流式调用（重试 2 次，最终失败抛 VisionFatalError）。

        messages 中可含 image_url 分段（OpenAI 兼容 content parts）。
        """
        client = self._ensure_client()
        params: dict[str, Any] = dict(
            model=self.route.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        if self.route.extra_body:
            params["extra_body"] = self.route.extra_body
        last_error: Exception | None = None
        for _attempt in range(1 + retries):
            try:
                resp = client.chat.completions.create(**params)
                content = ""
                if resp.choices:
                    choice = resp.choices[0]
                    msg = getattr(choice, "message", None)
                    content = (getattr(msg, "content", None) or "") if msg else ""
                if content is None:
                    content = ""
                self._report_usage(getattr(resp, "usage", None))
                return content
            except VisionFatalError:
                raise
            except Exception as exc:
                last_error = exc
        raise VisionFatalError(
            f"视觉模型调用失败（已重试 {retries} 次，provider="
            f"{self.route.provider}, model={self.route.model_name}）: {last_error}"
        )

    # ------------------------------------------------------------------
    # 图像消息构造
    # ------------------------------------------------------------------
    @staticmethod
    def image_content_part(
        image_path: Path | str,
        *,
        detail: str = "auto",
        base64_mode: str = "data_url",
        allow_external_url: bool = True,
    ) -> dict:
        """构造图像 content part（4.2）。

        - http(s) 地址：透传外部 URL，由服务端自行下载（零磁盘、请求体极小）；
          提供方未声明支持时抛 VisionFatalError，由调用方降级为本地落地
        - 本地文件：按真实后缀给出 MIME（原实现两个分支恒为 image/png，
          会把 .jpg/.webp 错标为 png）；base64_mode 决定是否带 data URL 前缀
          （部分 GLM 网关只接受裸 base64）
        """
        source = str(image_path).strip()
        if is_remote_url(source):
            if not allow_external_url:
                raise VisionFatalError(
                    f"当前视觉提供方未声明支持外部图片 URL（main_mlm.provider="
                    f"配置未开启 supports_external_url），请先把图片保存到本地: {source}"
                )
            return {
                "type": "image_url",
                "image_url": {"url": source, "detail": detail},
            }
        p = Path(image_path)
        try:
            payload = base64.b64encode(p.read_bytes()).decode("ascii")
        except OSError as exc:
            raise VisionFatalError(f"读取图像失败（视觉旁路输入）: {p} ({exc})") from exc
        mime = mime_for_path(p)
        url = payload if base64_mode == "raw" else f"data:{mime};base64,{payload}"
        return {
            "type": "image_url",
            "image_url": {"url": url, "detail": detail},
        }

    @staticmethod
    def image_message(prompt: str, image_path: Path | str, *, detail: str = "auto") -> list[dict]:
        """构造 [system, user(图+文)] 消息序列（视觉提取调用形态）。"""
        return [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    VisionClient.image_content_part(image_path, detail=detail),
                    {"type": "text", "text": prompt if not prompt else "请按 system 指令提取该图像信息"},
                ],
            },
        ]

    @staticmethod
    def multi_image_message(prompt: str, image_paths: list[Path | str], *, detail: str = "auto") -> list[dict]:
        """构造多图消息序列（图像段在前、指令在后，同 4.7 原理）。"""
        parts = [VisionClient.image_content_part(p, detail=detail) for p in image_paths]
        parts.append({"type": "text", "text": "请按 system 指令依次提取每张图像的信息，并输出图间差异对比"})
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------
    # 路由感知的消息构造（实例方法，自动套用 base64_mode / URL 能力）
    # ------------------------------------------------------------------
    def build_image_part(self, source: Path | str, *, detail: str = "auto") -> dict:
        """按当前路由能力构造单个图像段（本地路径或远程 URL）。"""
        return self.image_content_part(
            source,
            detail=detail,
            base64_mode=self.route.base64_mode,
            allow_external_url=self.route.supports_external_url,
        )

    def build_image_message(
        self, prompt: str, source: Path | str, *, detail: str = "auto"
    ) -> list[dict]:
        """单图消息序列（图像段在前、指令在后）。"""
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    self.build_image_part(source, detail=detail),
                    {"type": "text", "text": "请按 system 指令提取该图像信息"},
                ],
            },
        ]

    def build_multi_image_message(
        self, prompt: str, sources: list[Path | str], *, detail: str = "auto"
    ) -> list[dict]:
        """多图消息序列（每图独立图像段，顺序保持）。"""
        parts = [self.build_image_part(s, detail=detail) for s in sources]
        parts.append({"type": "text", "text": "请按 system 指令依次提取每张图像的信息，并输出图间差异对比"})
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": parts},
        ]
