"""L0 引用检测与输入分流（spec/multimodal_spec.md 3.1 / 3.6）。

职责（纯正则、零 LLM 成本）：
- @ 路径语法（显式引用）：`@D:/path/to/file.png`，含空格路径用引号
  `@"D:/my pics/a.png"`；可多个、穿插任意位置；仅支持绝对路径；
  解析后从 A 中剥离 @ 标记，A = 剩余文本 strip 后的结果
- 裸绝对路径与已知扩展名文件名引用（隐式候选）：复用现有"文件检测"
  规则，检测结果降级为候选引用，角色由 L1 裁定；裸路径在 A 中
  原样保留（不破坏用户原文）
- L0 不做任何格式与存在性拦截（不存在/不支持的路径也是候选，
  问题在后续环节暴露，3.6）

判定矩阵（3.3）：L0 零候选 -> 纯文字路径（零侵入，视觉/意图模块
零激活）；有候选 -> 交给 L1（intent_classifier）裁定角色。

本模块不含 LLM 调用，不抛会话终止类异常。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# 正则定义
# ----------------------------------------------------------------------
# @ 后跟引号包裹的绝对路径（支持含空格路径，3.6）
_AT_QUOTED = re.compile(r'@\s*"([^"\n]+)"')

# 路径字符集：排除空白与常见中英文标点界定符，允许 CJK 与普通文件名字符
_PATH_CHARS = r'[^\s，。；、！？）)】》」』"\'<>|*?]'

# @ 后跟裸 Windows 绝对路径（盘符开头，仅支持绝对路径，3.6）
_AT_BARE = re.compile(r'@([A-Za-z]:[\\/]' + _PATH_CHARS + r'*)')

# 裸 Windows 绝对路径候选（A 中原样保留，仅记录候选）
_ABS_WIN = re.compile(r'([A-Za-z]:[\\/]' + _PATH_CHARS + r'*)')

# 默认文本/代码扩展名（裸文件名候选，"按 521.txt 修改"类）；
# 媒体扩展名由调用方从 registry.media_extensions 动态传入合并（D10）
_KNOWN_TEXT_EXTS = (
    "md", "txt", "py", "yaml", "yml", "json", "toml", "ini", "cfg",
    "html", "css", "js", "ts", "xml", "csv", "log", "bat", "ps1",
    "rs", "go", "java", "c", "cpp", "h", "hpp", "sh", "sql",
)

# 文件名字符（含 CJK 与点号，首字符不为点）
_FNAME_CHARS = r'[A-Za-z0-9_\u4e00-\u9fff.\-]'

# 匹配末尾需剥离的标点（句读/括号/引号收尾等）
_TRAILING_PUNCT = '.,;:!?)]}）】》」』"\'、。；，：'


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class FileRef:
    """L0 检测出的候选引用（角色未定，由 L1 裁定）。"""

    path: str               # 规范化后的绝对路径或文件名（正斜杠）
    explicit: bool = False  # True=@ 显式引用；False=裸路径/文件名隐式候选
    start: int = -1         # 命中位置（@ 引用为原文位置；裸候选为 clean_text 位置）
    end: int = -1

    @property
    def via_at(self) -> bool:
        return self.explicit


@dataclass
class ResolvedInput:
    """L0 分流结果（3.1）。

    - clean_text：@ 标记剥离后的用户文本（隐式候选的裸路径保留其中）
    - references：候选引用清单（去重、保序）
    - 引用为空 -> 纯文字路径，L1/视觉/预读全部不激活（零侵入，3.4）
    """

    raw_text: str
    clean_text: str
    references: list[FileRef] = field(default_factory=list)

    @property
    def has_references(self) -> bool:
        return bool(self.references)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def normalize_path(raw: str) -> str:
    """路径规范化：反斜杠转正斜杠，去首尾空白与包裹引号。"""
    s = (raw or "").strip().strip('"').strip("'").strip()
    return s.replace("\\", "/")


def _strip_trailing_punct(path: str) -> str:
    """剥离路径末尾粘连的句读标点（如句号、右括号）。"""
    return path.rstrip(_TRAILING_PUNCT)


def _filename_pattern(media_extensions=None) -> re.Pattern:
    """构造已知扩展名文件名匹配（文本扩展名 + 媒体扩展名合并，大小写不敏感）。"""
    exts = set(_KNOWN_TEXT_EXTS)
    for ext in media_extensions or []:
        e = str(ext).lstrip(".").strip().lower()
        if e:
            exts.add(e)
    ext_alt = "|".join(re.escape(e) for e in sorted(exts))
    return re.compile(
        r'[A-Za-z0-9_\u4e00-\u9fff]' + _FNAME_CHARS + r'*\.(?:' + ext_alt + r')\b',
        re.IGNORECASE,
    )


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def resolve_input(
    text: str,
    media_extensions: list[str] | None = None,
) -> ResolvedInput:
    """L0 引用检测与分流（3.1 总流程的第一步）。

    - media_extensions：已启用媒体的扩展名清单（来自 registry，D10
      配置驱动），并入裸文件名候选的已知扩展名集合
    - 返回 ResolvedInput；零候选时 has_references=False（纯文字路径，
      零侵入：调用方不得触发 L1/视觉/预读）
    """
    raw = text or ""
    references: list[FileRef] = []
    seen: set[str] = set()

    def _add(path: str, explicit: bool, start: int, end: int) -> None:
        p = normalize_path(path)
        if not p or p in seen:
            return
        seen.add(p)
        references.append(FileRef(path=p, explicit=explicit, start=start, end=end))

    # 1) @ 显式引用：先取引号形式（可含空格），再取裸形式（跳过重叠区间）
    quoted_spans: list[tuple[int, int]] = []
    at_matches: list[tuple[int, int, str]] = []  # (start, end, path)
    for m in _AT_QUOTED.finditer(raw):
        span = (m.start(), m.end())
        quoted_spans.append(span)
        at_matches.append((m.start(), m.end(), m.group(1)))
    for m in _AT_BARE.finditer(raw):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, q) for q in quoted_spans):
            continue
        at_matches.append((m.start(), m.end(), m.group(1)))
    at_matches.sort(key=lambda t: t[0])

    # 2) 从原文剥离 @ 引用区间 -> clean_text
    parts: list[str] = []
    cursor = 0
    for start, end, _ in at_matches:
        parts.append(raw[cursor:start])
        cursor = end
    parts.append(raw[cursor:])
    clean_text = "".join(parts).strip()

    # 3) @ 引用登记（角色由 L1 裁定；L0 不做存在性/格式拦截，3.6）
    for start, end, path in at_matches:
        _add(_strip_trailing_punct(path), True, start, end)

    # 4) 裸绝对路径候选（clean_text 中扫描；A 中原样保留，不剥离）
    bare_spans: list[tuple[int, int]] = []
    for m in _ABS_WIN.finditer(clean_text):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, b) for b in bare_spans):
            continue
        bare_spans.append(span)
        _add(_strip_trailing_punct(m.group(1)), False, span[0], span[1])

    # 5) 已知扩展名文件名候选（跳过与裸绝对路径重叠的尾段匹配）
    fname_re = _filename_pattern(media_extensions)
    for m in fname_re.finditer(clean_text):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, b) for b in bare_spans):
            continue
        # 前一字符为路径分隔符说明是更长路径的尾段（未被 4 覆盖的残段），
        # 仅当该残段本身形如绝对路径时才保留；此处统一放行为候选
        _add(_strip_trailing_punct(m.group(0)), False, span[0], span[1])

    return ResolvedInput(raw_text=raw, clean_text=clean_text, references=references)
