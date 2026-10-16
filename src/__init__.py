"""兼容 shim（设计文档 9.1 Phase 1）：旧 `src.X` 导入路径过渡期转发到 `wdclaude.X`。

背景：包结构已重组为单一顶层包 `wdclaude/`（src/wdclaude/）。
本 shim 支持 `from src import X` / `src.X.属性` 形式的残留引用（PEP 562），
过渡一个版本后删除（届时全仓应仅使用 `from wdclaude.X import ...`）。

注意：`src.A2A`、`src.chicken_farm` 不再提供（分别移至 dev_only/A2A、已删除）。
"""
import importlib

_forwarded = {
    'cli', 'query', 'utility', 'llm_tool',
    'memory_ex', 'command', 'tools',
}


def __getattr__(name):
    if name in _forwarded:
        return importlib.import_module(f'wdclaude.{name}')
    raise AttributeError(f"module 'src' has no attribute '{name}'")
