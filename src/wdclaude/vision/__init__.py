"""WDClaude 多模态视觉模块入口（spec/multimodal_spec.md 第 9 章模块规划）。

提供模块级门面：
- MediaKind / MediaAsset / MediaHandler / MediaRegistry（媒体抽象，第 8 章）
- VisionFatalError / IntentClassifyFatalError（fail-fast 异常族，4.9）
- 关键门面函数：
  - create_vision_runtime(...)：从全局配置一次性装配视觉运行时
    （registry + client + extractor + view_image 工具，会话级单例）
  - resolve_input(...)：L0 引用检测（无引用输入零激活，3.4）
  - classify(...)：L1 意图分类（simple_chat 旁路单轮调用，3.2）
  - run_vision_pipeline(...)：级联方案 A+B 流水线（4.1）

零侵入承诺（D4 / 3.4）：query_loop 在 L0 检出零候选时不得调用本模块
任何函数（create_vision_runtime 也应惰性装配，仅在首条含引用输入时构建）。
"""

from __future__ import annotations

from typing import Any, Callable

from wdclaude.vision.image_processor import ImageDecodeError
from wdclaude.vision.input_resolver import FileRef, ResolvedInput, resolve_input
from wdclaude.vision.intent_classifier import (
    IntentClassifyFatalError,
    IntentResult,
    classify,
)
from wdclaude.vision.media_types import (
    MediaAsset,
    MediaHandler,
    MediaKind,
    MediaRegistry,
    MediaRegistryError,
)
from wdclaude.vision.vision_client import (
    MlmRoute,
    VisionClient,
    VisionFatalError,
    resolve_mlm_route,
)
from wdclaude.vision.vision_extractor import (
    ImageExtraction,
    UncertainRegion,
    VisionExtractionResult,
    VisionExtractor,
    run_vision_pipeline,
)

__all__ = [
    "MediaKind",
    "MediaAsset",
    "MediaHandler",
    "MediaRegistry",
    "MediaRegistryError",
    "VisionClient",
    "VisionFatalError",
    "MlmRoute",
    "resolve_mlm_route",
    "VisionExtractor",
    "VisionExtractionResult",
    "ImageExtraction",
    "UncertainRegion",
    "run_vision_pipeline",
    "IntentClassifyFatalError",
    "IntentResult",
    "classify",
    "resolve_input",
    "ResolvedInput",
    "FileRef",
    "ImageDecodeError",
    "create_vision_runtime",
    "VisionRuntime",
    "get_vision_runtime",
    "reset_vision_runtime",
]


class VisionRuntime:
    """视觉运行时（会话级）：registry + client + extractor + view_image 工具。

    由 create_vision_runtime 从全局配置装配；装配失败（配置缺失、
    handler_class 加载失败等）抛 MediaRegistryError / VisionFatalError，
    会话初始化阶段即报错，不拖到运行中（spec 8.1 / 4.9）。
    """

    def __init__(
        self,
        registry: MediaRegistry,
        client: VisionClient,
        extractor: VisionExtractor,
        *,
        mode: str = "cascade",
        max_zoom: int,
        token_budget: int,
        vision_enabled: bool = True,
        intent_classify_enabled: bool = True,
    ) -> None:
        self.registry = registry
        self.client = client
        self.extractor = extractor
        self.mode = "native" if mode == "native" else "cascade"
        self.max_zoom = int(max_zoom)
        self.token_budget = int(token_budget)
        self.vision_enabled = vision_enabled
        self.intent_classify_enabled = intent_classify_enabled
        # view_image 工具惰性创建（tool_executor 注册时拉起）
        self._view_image: Any | None = None

    @property
    def view_image(self):
        """view_image 工具实例（惰性创建，捕获本会话熔断状态）。"""
        if self._view_image is None:
            from wdclaude.tools.view_image import ViewImageTool

            self._view_image = ViewImageTool(
                self.registry,
                self.extractor,
                mode=self.mode,
                max_zoom=self.max_zoom,
                token_budget=self.token_budget,
            )
        return self._view_image


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_vision_runtime(
    global_cfg: Any,
    *,
    on_usage: Callable[[str, str, int, int], None] | None = None,
) -> VisionRuntime:
    """从全局配置装配视觉运行时（会话级单例）。

    参数：
    - global_cfg：config_loader 产出的全局配置（SimpleNamespace 点号访问，
      global_cfg.vision / global_cfg.media / global_cfg.multimodal /
      model_key 配置经 global_cfg.model 或 global_cfg.model_key 提供，
      视 config_loader 实际装载结构而定，两处均尝试读取）
    - on_usage：token 统计回调（注入 token_statistics 独立 vision 分类）

    读取的配置节（spec 10.1 / 10.2）：
    - vision：enabled / max_images_per_request / max_long_edge_px / detail /
      zoom_min_edge / view_image_max_zoom / view_image_token_budget /
      intent_classify_enabled
    - media.handlers：Handler 装配表（D10 配置驱动）
    - multimodal.mode：cascade | native（D2，默认 cascade）
    """
    vision_cfg = _cfg_get(global_cfg, "vision")
    media_cfg = _cfg_get(global_cfg, "media")
    multimodal_cfg = _cfg_get(global_cfg, "multimodal")

    registry = MediaRegistry.from_config(media_cfg)

    # model_key 配置定位：config_loader 将 model_key.yaml 内容合并至
    # global_cfg 顶层（main_mlm / 服务商节直接可达，含连字符节名经
    # getattr 访问）；兼容显式 model_key 子节的装载形态
    model_key_cfg = _cfg_get(global_cfg, "model_key")
    if model_key_cfg is None:
        model_key_cfg = global_cfg
    route = resolve_mlm_route(model_key_cfg)

    client = VisionClient(route)
    if on_usage is not None:
        client.on_usage = on_usage

    extractor = VisionExtractor(
        client,
        detail=str(_cfg_get(vision_cfg, "detail", "auto") or "auto"),
        max_images_per_request=int(_cfg_get(vision_cfg, "max_images_per_request", 5) or 5),
    )

    return VisionRuntime(
        registry,
        client,
        extractor,
        mode=str(_cfg_get(multimodal_cfg, "mode", "cascade") or "cascade"),
        max_zoom=int(_cfg_get(vision_cfg, "view_image_max_zoom", 5) or 5),
        token_budget=int(_cfg_get(vision_cfg, "view_image_token_budget", 50000) or 50000),
        vision_enabled=bool(_cfg_get(vision_cfg, "enabled", True)),
        intent_classify_enabled=bool(
            _cfg_get(vision_cfg, "intent_classify_enabled", True)
        ),
    )


# ----------------------------------------------------------------------
# 进程级运行时单例（spec 11 章：query_loop 输入分流与 tool_executor 的
# view_image 工具共享同一实例——main_mlm 客户端复用，view_image 会话级
# 熔断状态（zoom 次数 / token 预算）在同一进程内统一计数）
# ----------------------------------------------------------------------
_VISION_RUNTIME_SINGLETON: "VisionRuntime | None" = None


def get_vision_runtime() -> "VisionRuntime":
    """获取（或惰性装配）进程级视觉运行时单例。

    - 装配失败（配置缺失 / handler_class 错误）抛
      MediaRegistryError / VisionFatalError，由调用方处理
    - token 消耗统一注入 token_statistics 的 vision 分类
    """
    global _VISION_RUNTIME_SINGLETON
    if _VISION_RUNTIME_SINGLETON is None:
        from wdclaude.utility.config_loader import global_cfg
        from wdclaude.utility.token_statistics import record_token_usage

        def _on_usage(category: str, model: str, pt: int, ct: int) -> None:
            record_token_usage(
                model_name=model, prompt_tokens=pt, cached_tokens=0,
                completion_tokens=ct, query="(vision)", turn="vision-bypass",
            )

        _VISION_RUNTIME_SINGLETON = create_vision_runtime(global_cfg, on_usage=_on_usage)
    return _VISION_RUNTIME_SINGLETON


def reset_vision_runtime() -> None:
    """重置视觉运行时单例（新 Session 时调用：熔断计数归零，7.4 会话级语义）。"""
    global _VISION_RUNTIME_SINGLETON
    _VISION_RUNTIME_SINGLETON = None
