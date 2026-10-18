# WDClaude

Claude Code 的 Python 复刻——基于 LLM 的终端 AI 编程助手（由 MyClaude 重构而来）。

## 特性

- **XML 工具协议**：LLM 通过 `file_view / create / str_replace / bash / web_search / web_fetch / use_skill` 等标签驱动工具执行，多轮循环直至任务完成
- **启动即项目**：在哪里启动，哪里就是项目目录（B）；全部运行数据收纳于 `B/.wdclaude/`，不污染项目根
- **联网搜索与网页抓取**：配置驱动的 tools 协议实现（`model_key.yaml` 的 `web_search.provider` 引用命名模型块 + `search_api / fetch_api` 端点声明，代码不硬编码厂商端点）；富字段搜索结果（rerank_score / summary / authority 等）+ 单 URL 抓取返回 Markdown（`web_fetch`）；三层智能触发守卫（sys_prompt 判据 + 隐私/内网/限流硬守卫 + 缓存/预算经济层）；CLI Rich 面板与 LLM 上下文注入双通道分离
- **记忆系统**（memory_ex）：会话记忆提取 / 向量召回（FAISS）/ LLM 精排 / Bug 库
- **A2A 测试链**：单元/系统测试用例生成→沙箱执行→LLM 评判→Excel 报告（`/test` 系列命令）
- **双级配置**：包内出厂默认 + 项目级 `.wdclaude/config/` + 环境变量，密钥三级查找

## 快速开始

```bash
# 开发态（仓库内）
start.bat                        # 或 set PYTHONPATH=src && python -m wdclaude

# 安装态（任意目录）
pip install wdclaude-0.2.0-py3-none-any.whl
wdclaude                          # 交互模式
```

首次在项目目录启动会自动初始化 `.wdclaude/` 全套骨架，并引导填写 API Key。

## 常用命令

| 命令 | 用途 |
|------|------|
| `wdclaude` | 交互模式 |
| `wdclaude --version` | 版本 |
| `wdclaude doctor` | 环境体检 |
| `wdclaude --test-mode --prompt "..."` | 测试模式（结构化 JSON 输出） |
| `/test --ut-c / --st-c ...`（CLI 内） | A2A 测试链 |

## 可选依赖

```bash
pip install wdclaude[memory]   # numpy / faiss-cpu / openpyxl（向量记忆 + Excel 报表）
pip install wdclaude[search]   # requests / httpx（联网搜索）
pip install wdclaude[test]     # fastapi / uvicorn / pytest（A2A 测试链）
```

## 目录模型（v2.0）

```
<你的项目目录>/            # = 启动目录 B
├── .wdclaude/             # wdclaude 全部运行数据
│   ├── config/            #   config.yaml、model_key.yaml（密钥与 web_search 端点，不入 git）
│   ├── tmp_output/        #   AI 临时代码输出（探索/测试/草稿）
│   ├── log/sessions/      #   运行日志 + 会话日志（含 web_search/web_fetch 缓存）
│   ├── memory_storage/    #   记忆 / Bug 库
│   ├── skills/            #   项目技能
│   └── commands/          #   斜杠命令
└── spec/                  # 需求规格（项目资产）
```

## 安全提示

克隆他人项目时请注意：其 `.wdclaude/commands/`、`.wdclaude/skills/` 中可能包含指令模板，属于信任边界问题，请自行审查后再使用。

## License

见 [LICENSE](LICENSE)。
