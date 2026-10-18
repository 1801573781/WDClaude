---
name: "MK-TPL"
description: 根据用户 model_key.yaml 的改动同步更新 model_key.yaml.tpl 模板
category: Config
tags: [config, template, model_key, sync]
---

# MK-TPL：同步 model_key.yaml 到模板

根据用户实际使用的 `model_key.yaml`，同步修改模板 `model_key.yaml.tpl`，使模板的服务商节、模型参数与路由配置与用户配置保持一致。

**密钥红线**：模板中绝不写入任何真实密钥，所有 `api_key` 一律为 `"***"` 占位。

## 输入文件

- 用户配置（改动来源）：`D:/AI/WDClaude/.wdclaude/config/model_key.yaml`
- 模板（修改目标）：`D:/AI/WDClaude/src/wdclaude/defaults/model_key.yaml.tpl`

若用户在命令后指定了其他路径（如 `/mk-tpl <路径>`），以用户指定为准。

## 执行步骤

1. **读取用户配置**：用 `file_view` 完整读取 `model_key.yaml`，掌握全部服务商节与路由节。

2. **读取模板**：用 `file_view` 完整读取 `model_key.yaml.tpl` 当前内容。

3. **对比差异**，逐项识别以下变更：
   - 服务商节的增、删、改名（例如 `DeepSeek` 改为 `DeepSeek-main`）
   - 各节的 `base_url` / `model_name` / `extra_body` 变化
   - `web_search` 段的结构变化（provider 引用方式、`search_api` / `fetch_api` 等）
   - `main_model` / `main_mlm` / `memory.*` / `simple_chat` 的路由指向变化

4. **同步模板**：用 `str_replace` 逐处修改模板，遵守：
   - 模板头部注释、使用说明、结构说明保持不变；注释中的节名示例（如 `GPT / GLM-main / DeepSeek-main ...`）需与服务商节名保持一致
   - 新增节沿用模板的注释风格（节上方一行中文说明）
   - 所有 `api_key` 用 `"***"`，严禁复制真实密钥
   - 未变化的节一律不动

5. **输出摘要**：汇报本次同步内容（哪些节增删改、哪些模型名或路由变化）。

## 注意

- 只同步结构与参数，不同步密钥。
- `str_replace` 的 `<old>` 必须是 `file_view` 返回内容的原文复制，匹配失败时重新 `file_view` 再改。
- 同步完成后输出 `<done>` 结束任务。
