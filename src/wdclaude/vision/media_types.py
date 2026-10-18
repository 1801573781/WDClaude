"""媒体类型抽象与配置驱动的 Handler 装配器（spec/multimodal_spec.md 第 8 章，决策 D10）。

- MediaKind / MediaAsset / MediaHandler 构成媒体抽象基座（视频/音频为 Phase 4 预留接口）
- kind -> Handler 的映射、启用开关、扩展名清单全部由 config.yaml 的 media.handlers
  段定义，装配时逐条按 handler_class 执行 importlib 动态加载，严禁在代码中以
  硬编码字典维护该映射
- 新增媒体类型 = 实现 Handler 子类（src/wdclaude/vision/handlers/ 包）+ 在
  media.handlers 增加一段配置，框架代码零改动，无 OS 相关绑定
- 装配失败（类路径错误、类不存在、未实现抽象方法）在初始化装配阶段即报错
  终止，不拖到运行中（spec 8.1）

配置节构形（spec 10.1）：

    media:
      handlers:
        image:
          enabled: true
          handler_class: "src.wdclaude.vision.handlers.image_handler.ImageHandler"
          extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """从 SimpleNamespace 或 dict 中读取配置项（两种形态等价支持）。

    视觉模块的所有配置读取统一经由本函数：兼容 config_loader 产出的
    SimpleNamespace 与原生 dict/YAML 映射，缺失键回退 default。
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class MediaKind(str, Enum):
    """媒体大类。VIDEO / AUDIO 为预留（Phase 4，本次不实现）。"""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class MediaAsset:
    """磁盘媒体资产（图像永不离盘，D3：上下文只持有本结构的文本视图）。"""

    path: Path                                # 磁盘绝对路径
    kind: MediaKind
    mime: str                                 # 如 "image/png"
    size_bytes: int
    dimensions: str = ""                      # 图像类 "1920x1080"，其他为空
    source: str = "user_at"                   # "user_at" | "clipboard" | "view_image"
    meta: dict = field(default_factory=dict)  # 原始路径等附加信息


class MediaHandler(ABC):
    """媒体 Handler 抽象基类。

    扩展名/mime 清单不在类中声明，统一由 config.yaml 的
    media.handlers.<kind>.extensions 提供（单一事实来源，D10）。
    """

    kind: ClassVar[MediaKind]

    @abstractmethod
    def resolve(self, ref: str, *, source: str = "user_at") -> MediaAsset:
        """解析引用为媒体资产（存在性/魔数/解码头部校验，不含预处理）。"""

    @abstractmethod
    def preprocess(self, asset: MediaAsset) -> MediaAsset:
        """预处理（校验+压缩等），产物落盘、不覆盖原图，返回新 MediaAsset。"""

    @abstractmethod
    def to_text_view(self, asset: MediaAsset, focus: str | None = None) -> str:
        """级联方案的资产文本视图（B 的基础来源，D5 三要素）。"""

    @abstractmethod
    def to_content_part(self, asset: MediaAsset) -> dict:
        """native 方案的 OpenAI 兼容 content part（图像段）。"""


class MediaRegistryError(RuntimeError):
    """媒体 Handler 装配失败：初始化装配阶段即抛出终止（spec 8.1）。"""


class MediaRegistry:
    """配置驱动的媒体 Handler 注册表（D10）。

    经 from_config 装配：仅 enabled: true 的 handler 参与装配；
    扩展名 -> kind 的归类同样来自配置（含未启用项，供 fail-fast 提示
    定位配置位置）。
    """

    def __init__(
        self,
        handlers: dict[MediaKind, MediaHandler],
        ext_kind_all: dict[str, MediaKind],
        disabled_kinds: set[str],
    ) -> None:
        self._handlers = dict(handlers)
        self._ext_kind_all = {str(ext).lower(): kind for ext, kind in ext_kind_all.items()}
        self._disabled = set(disabled_kinds)

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, media_cfg: Any) -> "MediaRegistry":
        """从 config 的 media 段装配（media_cfg 为 SimpleNamespace 或 dict）。"""
        handlers: dict[MediaKind, MediaHandler] = {}
        ext_kind_all: dict[str, MediaKind] = {}
        disabled: set[str] = set()

        handlers_cfg = cfg_get(media_cfg, "handlers")
        if not handlers_cfg:
            raise MediaRegistryError(
                "config.yaml 缺少 media.handlers 配置段（或该段为空），无法装配媒体处理器；"
                "请按 spec/multimodal_spec.md 10.1 补充 media.handlers 配置"
            )
        if isinstance(handlers_cfg, dict):
            items = handlers_cfg.items()
        else:
            items = vars(handlers_cfg).items()

        for kind_name, entry in items:
            if entry is None:  # 注释掉的预留段（yaml 解析为 None）
                continue
            try:
                kind = MediaKind(kind_name)
            except ValueError:
                raise MediaRegistryError(
                    f"media.handlers 中存在未知媒体类型 '{kind_name}'，"
                    f"合法值: {[k.value for k in MediaKind]}"
                ) from None
            cls._collect_extensions(entry, kind, ext_kind_all)
            if not bool(cfg_get(entry, "enabled", False)):
                disabled.add(kind_name)
                continue
            handlers[kind] = cls._load_handler(kind_name, entry)

        if not handlers:
            raise MediaRegistryError(
                "media.handlers 中没有任何 enabled: true 的媒体处理器，"
                "至少需要启用 image 才能处理媒体输入"
            )
        return cls(handlers, ext_kind_all, disabled)

    @staticmethod
    def _collect_extensions(entry: Any, kind: MediaKind, sink: dict[str, MediaKind]) -> None:
        exts = cfg_get(entry, "extensions", None) or []
        for ext in exts:
            sink[str(ext).lower()] = kind

    @staticmethod
    def _load_handler(kind_name: str, entry: Any) -> MediaHandler:
        class_path = cfg_get(entry, "handler_class", "")
        if not class_path or "." not in class_path:
            raise MediaRegistryError(
                f"media.handlers.{kind_name}.handler_class 缺失或格式错误: {class_path!r}，"
                f"应为 '模块路径.类名' 形式"
            )
        module_path, _, class_name = class_path.rpartition(".")
        # 双根回退：兼容 src 根运行（python -m src.wdclaude）与安装模式
        # （pip 安装后仅 wdclaude 可导入）两种导入根
        candidates = [module_path]
        if module_path.startswith("wdclaude."):
            candidates.append("src." + module_path)
        elif module_path.startswith("src.wdclaude."):
            candidates.append(module_path[len("src."):])
        handler_cls = None
        last_exc: Exception | None = None
        for mod in candidates:
            try:
                module = importlib.import_module(mod)
                handler_cls = getattr(module, class_name)
                break
            except Exception as exc:
                last_exc = exc
        if handler_cls is None:
            raise MediaRegistryError(
                f"media.handlers.{kind_name}.handler_class 加载失败: {class_path} ({last_exc})"
            )
        if not (isinstance(handler_cls, type) and issubclass(handler_cls, MediaHandler)):
            raise MediaRegistryError(
                f"media.handlers.{kind_name}.handler_class 指向的类不是 MediaHandler 子类: {class_path}"
            )
        try:
            handler = handler_cls()
        except TypeError as exc:
            # 含未实现抽象方法的类在实例化时即抛 TypeError（spec 8.1）
            raise MediaRegistryError(
                f"media.handlers.{kind_name}.handler_class 实例化失败（可能存在未实现的"
                f"抽象方法）: {class_path} ({exc})"
            ) from exc
        except Exception as exc:
            raise MediaRegistryError(
                f"media.handlers.{kind_name}.handler_class 实例化失败: {class_path} ({exc})"
            ) from exc
        if getattr(handler, "kind", None) is not MediaKind(kind_name):
            handler_kind = getattr(handler, "kind", None)
            raise MediaRegistryError(
                f"media.handlers.{kind_name} 装配的 Handler kind 属性为 "
                f"{getattr(handler_kind, 'value', handler_kind)!r}，与配置节名 {kind_name} 不一致"
            )
        return handler

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def handler_for(self, kind: MediaKind) -> MediaHandler | None:
        """已装配 kind 的 Handler；未启用/未装配返回 None。"""
        return self._handlers.get(kind)

    def resolve_kind(self, ext: str) -> MediaKind | None:
        """扩展名 -> MediaKind 归类（含未启用 kind，供错误提示定位配置位置）。"""
        return self._ext_kind_all.get(str(ext).lower())

    def is_kind_enabled(self, kind: MediaKind) -> bool:
        return kind in self._handlers

    @property
    def enabled_kinds(self) -> list[MediaKind]:
        return list(self._handlers.keys())

    @property
    def disabled_kind_names(self) -> set[str]:
        return set(self._disabled)

    @property
    def media_extensions(self) -> list[str]:
        """已启用 kind 的全部扩展名（供 L0 文件名检测合并，spec 3.6）。"""
        return [
            ext
            for ext, kind in self._ext_kind_all.items()
            if kind in self._handlers
        ]

    def describe(self) -> str:
        """装配结果摘要（诊断与启动日志用）。"""
        parts = []
        for kind in MediaKind:
            handler = self._handlers.get(kind)
            if handler is None:
                continue
            exts = sorted(e for e, k in self._ext_kind_all.items() if k == kind)
            parts.append(f"{kind.value}={type(handler).__name__}({', '.join(exts)})")
        return "MediaRegistry[" + "; ".join(parts) + "]"
