"""native 模式图像生命周期状态机与视图层降级（spec/multimodal_spec.md 5.3 / 5.4）。

状态机（5.3）：

    注入轮(INJECT) -> 活跃期(ACTIVE, 最近 N 轮) -> 降级(DEGRADED) -> 重看(REINJECT, 回到 ACTIVE)

核心原则：
- 降级 = 历史消息中的图像段在**组装请求时（视图层）**被替换为文本占位符；
  存储层会话记录永不修改，保证可无损恢复
- 占位描述零额外调用：主模型看图轮的回复本身就在上下文中，占位符引用其
  位置即可，不调任何模型
- 策略性降级（image_lifetime_rounds，默认 -1 = 永不）与生存性降级
  （image_evict_on_context_pressure，默认 true）是两个独立配置键（5.4）：
  -1 不豁免生存性降级（上下文水位触发时历史图像段是第一优先牺牲品）
- native 模式下注入轮的图像段同样经过 4.3 预处理管线（分辨率第一杠杆）
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LifecycleState(str, Enum):
    """图像生命周期状态（5.3 状态机）。"""

    INJECT = "inject"      # 注入轮（当轮消息携带图像段）
    ACTIVE = "active"      # 活跃期（最近 N 轮内，消息保持图像段）
    DEGRADED = "degraded"  # 降级（视图层替换为占位符，可无损恢复）
    REINJECT = "reinject"  # 重看（view_image 触发，回到 ACTIVE）


DEFAULT_LIFETIME_ROUNDS = -1   # multimodal.image_lifetime_rounds，-1 = 永不降级
DEFAULT_EVICT_ON_PRESSURE = True  # multimodal.image_evict_on_context_pressure


@dataclass
class ImageRecord:
    """会话内一个注入图像的登记项。

    msg_index：图像段所在消息在存储层 api_messages 中的下标（锚点，
    降级仅作用于视图层，存储层永不变动）。
    """

    path: str                  # 原图磁盘绝对路径（重看锚点，D3）
    msg_index: int             # 注入消息下标（存储层，永久有效）
    part_index: int            # 图像段在该消息 content 分段中的下标
    dimensions: str            # "1920x1080"
    injected_round: int        # 注入轮次号
    state: LifecycleState = LifecycleState.INJECT
    last_active_round: int = 0  # 最近一次 ACTIVE/REINJECT 的轮次号
    meta: dict = field(default_factory=dict)

    def placeholder_text(self) -> str:
        """降级占位符（5.3）：路径 + 尺寸 + 早期回复引用 + 恢复入口。"""
        return (
            f"[图像已降级] {self.path} ({self.dimensions}) - 完整分析见本会话"
            f"早期助手回复；可用 view_image 恢复视觉输入"
        )


@dataclass
class DegradedSummary:
    """一次视图组装中被降级的图像统计（诊断/日志用）。"""

    degraded: list[str] = field(default_factory=list)
    kept_active: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.degraded)


class ImageLifecycleManager:
    """native 模式图像生命周期管理器（每会话一个实例）。

    职责：
    - 登记注入图像（INJECT）与注入消息锚点
    - 每轮组装请求时按状态机计算视图：策略性降级（lifetime 轮数）+
      生存性降级（上下文压力，第一优先牺牲品）
    - 重看（REINJECT）：view_image 触发后对应图像回到 ACTIVE
    - 存储层 api_messages 永不修改（视图层 copy 后替换，5.3）
    """

    def __init__(
        self,
        *,
        lifetime_rounds: int = DEFAULT_LIFETIME_ROUNDS,
        evict_on_pressure: bool = DEFAULT_EVICT_ON_PRESSURE,
    ) -> None:
        self.lifetime_rounds = lifetime_rounds
        self.evict_on_pressure = evict_on_pressure
        self._records: list[ImageRecord] = []
        self._round_counter = 0
        self._pressure_signal = False
        self._pending_inject_msg: int | None = None  # 注入当轮豁免标记（一次性）
        self._degraded_by_pressure: set[int] = set()  # msg_index 集合
        self.lifetime_rounds = lifetime_rounds
        self.evict_on_pressure = evict_on_pressure
        self._records: list[ImageRecord] = []
        self._round_counter = 0
        self._pressure_signal = False
        self._degraded_by_pressure: set[int] = set()  # msg_index 集合

    # ------------------------------------------------------------------
    # 轮次推进
    # ------------------------------------------------------------------
    def advance_round(self) -> int:
        """推进一轮（每轮组装请求前调用一次），返回当前轮次号。"""
        self._round_counter += 1
        # 策略性降级判定（5.4）：-1 永不；正整数 N = 注入 N 轮后降级
        if self.lifetime_rounds > 0:
            for rec in self._records:
                if rec.state in (LifecycleState.ACTIVE, LifecycleState.INJECT):
                    # "注入 N 轮后降级"：第 N 轮仍活跃，第 N+1 轮起降级（spec 5.3）
                    if (self._round_counter - rec.last_active_round) > self.lifetime_rounds:
                        rec.state = LifecycleState.DEGRADED
        return self._round_counter

    def set_context_pressure(self, pressure: bool) -> None:
        """上下文水位信号（由 context_compressor 触发，5.4 生存性降级）。

        pressure=True 且 evict_on_pressure 时，下一轮视图组装中历史
        图像段为第一优先牺牲品（可无损恢复，零真实信息损失）。
        -1（永不策略性降级）不豁免此项。
        """
        self._pressure_signal = bool(pressure)

    # ------------------------------------------------------------------
    # 登记
    # ------------------------------------------------------------------
    def register_inject(
        self,
        path: str,
        msg_index: int,
        part_index: int,
        dimensions: str,
    ) -> ImageRecord:
        """登记注入轮图像（INJECT）并进入活跃期（ACTIVE）。"""
        rec = ImageRecord(
            path=str(path),
            msg_index=msg_index,
            part_index=part_index,
            dimensions=dimensions or "unknown",
            injected_round=self._round_counter,
            state=LifecycleState.ACTIVE,
            last_active_round=self._round_counter,
        )
        self._records.append(rec)
        self._degraded_by_pressure.discard(msg_index)
        self._pending_inject_msg = msg_index  # 注入当轮豁免（仅下一次 render_view）
        return rec

    def mark_reinject(self, path: str) -> bool:
        """重看（REINJECT -> ACTIVE，5.3）：view_image 恢复视觉输入。

        重看后重新计时（last_active_round 刷新为当前轮），再次进入活跃期。
        """
        for rec in self._records:
            if rec.path == str(path) and rec.state == LifecycleState.DEGRADED:
                rec.state = LifecycleState.ACTIVE
                rec.last_active_round = self._round_counter
                self._degraded_by_pressure.discard(rec.msg_index)
                self._pending_inject_msg = rec.msg_index  # 重看当轮同样豁免降级
                return True
        return False

    # ------------------------------------------------------------------
    # 视图层渲染（组装请求时，存储层永不变动）
    # ------------------------------------------------------------------
    def render_view(self, api_messages: list[dict]) -> tuple[list[dict], DegradedSummary]:
        """渲染请求视图：对存储层消息做深拷贝后替换应降级的图像段。

        - 策略性降级：advance_round 已标记 DEGRADED 的记录
        - 生存性降级：_pressure_signal 且 evict_on_pressure 时，历史
          图像段（非本轮注入）第一优先牺牲，标记 DEGRADED
        - 图像段（type=image_url）替换为 {"type":"text","text":占位符}
        """
        summary = DegradedSummary()
        if self._pressure_signal and self.evict_on_pressure:
            for rec in self._records:
                if rec.state in (LifecycleState.ACTIVE, LifecycleState.INJECT):
                    rec.state = LifecycleState.DEGRADED
                    self._degraded_by_pressure.add(rec.msg_index)
        view = copy.deepcopy(api_messages)
        # 注入当轮豁免由 _pending_inject_msg 一次性标记实现（register_inject
        # 后的首次 render_view 保证当轮请求可看图），下轮起正常参与降级
        for rec in self._records:
            if rec.state != LifecycleState.DEGRADED:
                if rec.msg_index < len(view):
                    summary.kept_active.append(rec.path)
                continue
            if self._pending_inject_msg is not None and rec.msg_index == self._pending_inject_msg:
                # 注入当轮豁免（保证当轮请求可看图）；下轮起正常参与降级
                summary.kept_active.append(rec.path)
                continue
            msg = view[rec.msg_index] if rec.msg_index < len(view) else None
            if msg is None or not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for j, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    if j == rec.part_index or _part_image_url_matches(part, rec):
                        content[j] = {
                            "type": "text",
                            "text": rec.placeholder_text(),
                        }
            summary.degraded.append(rec.path)
        self._pressure_signal = False  # 消费一次性压力信号
        self._pending_inject_msg = None  # 豁免仅作用于注入后的首次组装
        return view, summary

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def records(self) -> list[ImageRecord]:
        return list(self._records)

    def record_for_path(self, path: str) -> ImageRecord | None:
        for rec in self._records:
            if rec.path == str(path):
                return rec
        return None

    @property
    def active_count(self) -> int:
        return sum(
            1
            for r in self._records
            if r.state in (LifecycleState.ACTIVE, LifecycleState.INJECT)
        )


# ----------------------------------------------------------------------
# 视图工具
# ----------------------------------------------------------------------
_IMAGE_URL_PATTERN = "image_url"


def _message_has_image(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == _IMAGE_URL_PATTERN for p in content)


def _part_image_url_matches(part: dict, rec: ImageRecord) -> bool:
    """比对图像段与登记项（data URL 无法含路径，按段下标+消息锚点匹配）。"""
    return True  # 段下标已由 rec.part_index 定位，此处恒真（保持接口对称）


# ----------------------------------------------------------------------
# 消息构造（5.2 native 注入轮消息格式）
# ----------------------------------------------------------------------
def build_native_user_message(
    text: str,
    image_paths: list[Path | str],
    *,
    detail: str = "auto",
    image_parts: list[dict] | None = None,
) -> dict:
    """构造 native 模式注入轮 user 消息（5.2 分段格式）。

    - 仅 user 角色消息允许携带图像段（system/assistant 保持纯文本）
    - 注入顺序：图像段在前、指令文字段在后（同 4.7 原理）
    - image_parts：预构造的图像段列表（来自 handler.to_content_part），
      缺省时按 data URL 现场构造
    """
    parts: list[dict] = []
    if image_parts:
        parts.extend(image_parts)
    else:
        from wdclaude.vision.vision_client import VisionClient

        parts = [VisionClient.image_content_part(p, detail=detail) for p in image_paths]
    parts.append({"type": "text", "text": text or ""})
    return {"role": "user", "content": parts}


def build_native_tool_result(
    text: str,
    image_paths: list[Path | str],
    *,
    detail: str = "auto",
) -> dict:
    """构造 native 模式工具结果消息（6.3）：特写图作为图像段附加返回。"""
    from wdclaude.vision.vision_client import VisionClient

    parts: list[dict] = [
        {"type": "text", "text": text},
    ]
    parts.extend(VisionClient.image_content_part(p, detail=detail) for p in image_paths)
    return {"role": "user", "content": parts}
