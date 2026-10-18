# WDClaude — Agent Instructions

## Entry & Launch

- **Run (dev)**: `start.bat`（设 PYTHONPATH=src 后 `python -m wdclaude`）
- **Run (installed)**: pip 安装 wheel 后任意目录 `wdclaude`
- **CLI args**: `wdclaude [--role mycode] [--version] [doctor] [--test-mode --prompt "..." [--test-output x.json]]`
- **v2.0 原则**：启动位置即项目目录 B，无 --self/--project——修改哪里由用户指令中的路径决定
- **Proxy** (corp env): `start.bat` 中注释的代理行按需启用（GLM 国内直连，默认关闭）

## Config（两级 + 环境变量，v2.0 第 6 章）

优先级：环境变量 > B/.wdclaude/config > 包内 defaults
- `B/.wdclaude/config/config.yaml` — 唯一活跃配置（web_search 参数段：trigger_mode / sensitive_words / 预算 / 限流 / 会话上限等）
- `B/.wdclaude/config/model_key.yaml` — API 密钥（provider 顶层节 + main_model.provider 引用；web_search 段为 `provider 引用 + search_api / fetch_api 端点声明`；密钥红线，不入 git）
- 密钥三级查找：WDCLAUDE_API_KEY env → B 级 model_key.yaml → ~/.wdclaude/model_key.yaml 兜底
- `B/.wdclaude/config/memory/memory_ex.yaml` — 记忆后端配置
- All config merged into `SimpleNamespace` at `global_cfg = load_config()`（`wdclaude/utility/config_loader.py`）

## Directory Model（v2.0 启动即项目）

- **A1 包代码**: `src/wdclaude/`（开发态）/ site-packages（安装态），含 defaults/ 出厂默认
- **B 项目目录** = cwd；全部运行数据在 `B/.wdclaude/` 下：
  config/ tmp_output/ context/ data/ docs/ log/(含 sessions/) logo/ memory_storage/ skills/ tests/ token_statistics/ backups/ cache/ commands/
- `B/spec/` 为项目资产留 B 根；`B/.tree_cache.md` 为 /init 工程树缓存

## Testing & Linting

- **Test**: `pytest`（pytest.ini：testpaths = src .wdclaude/tests）
- **Lint**: `ruff check .`（ruff.toml）
- **A2A 测试链**: `/test --ut-c/--ut-e/--ut-a2a/--st-c/--st-e/--st-a2a`（MyOrch:8200, SystemTest:8201, UnitTest:8202）

## Architecture

- **Sync only** — no async/await in core loop. `stream_chat()` returns `(content, is_truncated, reasoning_content)`.
- **Flow**: `__main__.py → MyClaudeCLI → QueryLoop.run() → chat_llm → parse_tools → execute_tools → loop`
- **Display decoupling**: `QueryLoop` takes callbacks from CLI. No direct `console.print()` in business logic.
- **Memory backend**: `memory_ex`（factory 降级 NoopMemory）；记忆库固定 `B/.wdclaude/memory_storage/memory_ex/`
- **Session logs**: `B/.wdclaude/log/sessions/`（HTML+MD）
- **HTTP 强制直连**: chat_llm 三个 httpx 客户端 trust_env=False（防失效代理劫持）

## LLM Tool Protocol (XML-based)

...

## Critical Rules (from sys_prompt)

1. **Tools and `<done>` must be in SEPARATE turns**
2. **All file paths must be absolute**
3. **`summary` attribute required** on `<create>` and `<str_replace>` (≤50 chars)
4. **Never overwrite existing files** — use `file_view → str_replace` instead
5. **`<str_replace>` before file_view is forbidden** — `<old>` must be verbatim from prior `file_view`
6. **`<new>` must close with `</new>`**
7. **No `role="system"` mid-conversation** — MiniMax API rejects it
8. **`<done>` regex is lenient**: `(?:</done>|$)`
9. **Tool results are dicts**, never lists
10. **Windows native only** — no `ls`/`grep`/`rm`/`curl`; use `dir`/`findstr`/`del`

## A2A Subsystem（v2.0 已转正）

- `src/wdclaude/A2A/` — A2A 协议正式功能：MyOrch 编排 + SystemTest/UnitTest 执行服务 + LLM Judge + Docker/本地沙箱
- 随包分发，重依赖走 `pip install wdclaude[test]`（fastapi/uvicorn/pytest 等）
- 启动：`python -m wdclaude.A2A.myorch.main` 等

## Git

- `origin` → Gitee (lzb_ai/WDClaude)
- `.wdclaude/` 下运行数据（log/memory_storage/token_statistics/tmp_output/backups）与 model_key.yaml 均 gitignored
