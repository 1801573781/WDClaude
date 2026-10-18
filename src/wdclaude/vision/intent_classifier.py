"""L1 意图判断（spec/multimodal_spec.md 3.1 / 3.2）。

职责：
- 输入为 L0 候选清单 + 用户原始文本，经 simple_chat 路由单轮旁路调用
  （D6：与 embedding_client 同级的旁路服务，不进入主 query loop 流式通道）
- 输出 JSON：每个引用的角色分类（requirement_doc / operation_target /
  media_input）+ is_multimodal + summary
- 调用失败或输出解析失败：重试 2 次，仍失败则抛 IntentClassifyFatalError。
  调用方（query_loop）应捕获并按扩展名规则降级分类，而非终止用户输入
  （原 3.2 / 4.9 "宁停不猜" fail-fast 已废止：分类是便利性优化，降级优于丢弃整轮输入）

配置（vision.intent_classify_enabled，10.1）：
- true（默认）：L1 正常分类
- false：回退旧行为——候选引用一律按需求文档预读（本模块不激活，
  由调用方处理回退路径，本类提供 legacy 路径构建器）

prompt 位于 vision/prompts/intent_classify.txt（4.4 模块规划）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
INTENT_PROMPT_FILE = PROMPTS_DIR / "intent_classify.txt"

# 合法角色集
VALID_ROLES = ("requirement_doc", "operation_target", "media_input")
MAX_RETRIES = 2  # 重试 2 次（3.2）


class IntentClassifyFatalError(RuntimeError):
    """L1 意图判断最终失败（重试耗尽或输出无法解析）-> 终止会话（4.9）。"""


@dataclass
class RefRole:
    """单个引用的角色裁定结果。"""

    path: str
    role: str  # requirement_doc | operation_target | media_input


@dataclass
class IntentResult:
    """L1 分类结果（3.2 输出结构）。"""

    references: list[RefRole] = field(default_factory=list)
    is_multimodal: bool = False
    summary: str = ""

    @property
    def media_refs(self) -> list[RefRole]:
        return [r for r in self.references if r.role == "media_input"]

    @property
    def requirement_refs(self) -> list[RefRole]:
        return [r for r in self.references if r.role == "requirement_doc"]

    @property
    def operation_refs(self) -> list[RefRole]:
        return [r for r in self.references if r.role == "operation_target"]

    def role_list_text(self) -> str:
        """引用角色清单文本（记忆召回查询增强用，3.1）。"""
        if not self.references:
            return ""
        return "; ".join(f"{r.path}({r.role})" for r in self.references)


def load_prompt() -> str:
    """加载 L1 分类 prompt 模板。"""
    return INTENT_PROMPT_FILE.read_text(encoding="utf-8")


def build_user_message(text: str, candidates: list[str]) -> str:
    """构造 L1 调用的 user 消息：候选清单 + 用户原始文本。"""
    lines = ["候选引用清单（L0 检出，角色待定）:"]
    for i, p in enumerate(candidates, 1):
        lines.append(f"{i}. {p}")
    lines.append("")
    lines.append("用户原始输入:")
    lines.append(text if text.strip() else "(空)")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象（容忍围栏、前后杂文与未转义反斜杠）。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise IntentClassifyFatalError(
            f"L1 输出中未找到 JSON 对象: {text[:200]!r}"
        )
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 修复：Windows 路径反斜杠未转义（高频失败根因）——模型回显
    # "D:\AI\..." 时 \A \W 等为 JSON 非法转义序列。将后面不跟合法转义字符
    # （" \ / b f n r t u）的反斜杠补转义为 \\ 后重试
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise IntentClassifyFatalError(f"L1 输出 JSON 解析失败: {exc}") from exc


def parse_result(raw: str, candidates: list[str]) -> IntentResult:
    """解析 L1 输出为 IntentResult（路径对齐校验）。"""
    data = _extract_json(raw)
    refs_raw = data.get("references")
    if not isinstance(refs_raw, list):
        raise IntentClassifyFatalError("L1 输出缺少 references 数组")
    by_path = {normalize(p): p for p in candidates}
    refs: list[RefRole] = []
    for item in refs_raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        role = str(item.get("role", "")).strip().lower()
        if not path or role not in VALID_ROLES:
            continue
        # 路径对齐：模型输出路径须命中候选（规范化后比对），否则丢弃
        canonical = normalize(path)
        if canonical not in by_path:
            continue
        refs.append(RefRole(path=by_path[canonical], role=role))
    # 候选覆盖校验：3.2 要求 references 覆盖全部候选；漏判视为解析失败
    # （抛出后由调用方按扩展名规则降级分类，不再终止用户输入）
    missed = [p for p in candidates if normalize(p) not in {normalize(r.path) for r in refs}]
    if missed:
        raise IntentClassifyFatalError(
            f"L1 输出未覆盖全部候选引用（漏判: {missed}），拒绝猜测角色"
        )
    return IntentResult(
        references=refs,
        is_multimodal=bool(data.get("is_multimodal", False)) or any(
            r.role == "media_input" for r in refs
        ),
        summary=str(data.get("summary", "")).strip(),
    )


def normalize(path: str) -> str:
    """路径规范化（与 input_resolver.normalize_path 一致）。"""
    s = (path or "").strip().strip('"').strip("'").strip()
    return s.replace("\\", "/")


def classify(
    text: str,
    candidates: list[str],
    llm_call: Callable[[str], str],
    *,
    retries: int = MAX_RETRIES,
) -> IntentResult:
    """L1 意图分类主入口（重试 2 次，仍失败抛 IntentClassifyFatalError）。

    llm_call(prompt_user_message) -> 模型输出文本：由调用方注入
    simple_chat 旁路客户端的单轮调用（D6），本模块不直接依赖 OpenAI SDK。
    """
    if not candidates:
        return IntentResult()  # 无候选：L1 不激活（零侵入，3.4）
    system = load_prompt()
    user = build_user_message(text, candidates)
    last_error: Exception | None = None
    for _attempt in range(1 + retries):
        try:
            raw = llm_call(user)
            return parse_result(raw, candidates)
        except IntentClassifyFatalError as exc:
            last_error = exc  # 解析失败可重试（重试后重新调用模型）
        except Exception as exc:
            last_error = IntentClassifyFatalError(f"simple_chat 旁路调用失败: {exc}")
    raise IntentClassifyFatalError(
        f"L1 意图判断失败（已重试 {retries} 次）: {last_error}"
    )
