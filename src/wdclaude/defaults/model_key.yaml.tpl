# WDClaude API 密钥配置模板（密钥红线：本文件只含 *** 占位，绝不包含真实密钥）
#
# 使用方法：
#   1. 首次运行 wdclaude 时，本模板自动复制为 <项目目录>/.wdclaude/config/model_key.yaml
#   2. 编辑该文件，把 "***" 替换为你的真实 API Key（只需填写实际使用的服务商节）
#   3. 修改后重启生效；环境变量 WDCLAUDE_API_KEY 优先级更高
#   4. ~/.wdclaude/model_key.yaml 可作为全局兜底（多项目共用一份密钥时使用）
#
# 结构说明：
#   - 顶层各节（GPT / GLM-main / DeepSeek-main ...）= 服务商连接参数，节名即 provider 名，可按需增删
#   - main_model.provider  = 主模型（对话与编码）使用哪个 provider
#   - simple_chat.provider = 记忆提取/整理/进化、意图压缩等单轮调用使用哪个 provider
#   - main_mlm.provider    = 多模态模型（视觉提取/view_image/native 模式）使用哪个 provider
#   - memory.*             = 记忆系统的向量化与精排模型
#   - web_search           = 联网搜索后端（与 config.yaml 的 web_search 参数段配合）

# ===== 服务商连接参数（provider 节，按需增删）=====

# GPT API 连接参数
GPT:
  api_key: "***"
  base_url: "https://api.openai.com/v1"
  model_name: "gpt-5.5"

# GLM 主力模型（思考模式）
GLM-main:
  api_key: "***"
  base_url: "https://open.bigmodel.cn/api/coding/paas/v4"
  model_name: "glm-5.3"
  extra_body:
    thinking:
      type: "enabled"

# GLM 5.2（记忆精排等）
GLM-5.2:
  api_key: "***"
  base_url: "https://open.bigmodel.cn/api/coding/paas/v4"
  model_name: "glm-5.2"
  extra_body:
    thinking:
      type: "enabled"

# GLM Flash（轻量快速）
GLM-4-FLASH:
  api_key: "***"
  base_url: "https://open.bigmodel.cn/api/coding/paas/v4"
  model_name: "glm-4-flash"

# GLM Embedding（记忆向量化）
GLM-Embedding:
  api_key: "***"
  base_url: "https://open.bigmodel.cn/api/paas/v4/embeddings"
  model_name: "embedding-3"

# DeepSeek API 连接参数（DeepSeek 当前最先进模型）
DeepSeek-main:
  api_key: "***"
  base_url: "https://api.deepseek.com"
  model_name: "deepseek-flash"

# DeepSeek Flash API 连接参数
DeepSeek-Flash:
  api_key: "***"
  base_url: "https://api.deepseek.com"
  model_name: "deepseek-flash"

# MiniMax API 连接参数
MiniMax:
  api_key: "***"
  base_url: "https://api.minimaxi.com/v1"
  model_name: "MiniMax-M2.7"

# Qwen API 连接参数
Qwen-2.5:
  api_key: "***"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model_name: "qwen2.5-7b-instruct"

# GLM 多模态模型（main_mlm 路由，视觉提取 / view_image / native 多模态模式）
GLM-main-mlm:
  api_key: "***"
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  model_name: "glm-5.3-flashx"
  extra_body:
    thinking:
      type: "enabled"

# ===== 互联网搜索工具后端 =====
# DeepSeek 提供了 web_search 接口：provider 直接引用上方服务商节（复用其 api_key），
# search_api / fetch_api 分别为搜索与网页抓取接口
web_search:
  provider: "DeepSeek-main"
  search_api: "POST https://api.deepseek.com/v1/tools/search"
  fetch_api: "POST https://api.deepseek.com/v1/tools/fetch"

# ===== 主模型选择（provider 填写上方任一节名）=====
main_model:
  provider: "GLM-main"

# ===== 多模态模型选择（provider 填写上方任一节名）=====
# 用途：视觉提取（级联方案的 B 生产者）、view_image 执行、native 多模态模式下的主循环 LLM
# 两级路由：main_mlm.provider -> 服务商节（api_key / base_url / model_name / extra_body）
main_mlm:
  provider: "GLM-main-mlm"

# ===== 记忆系统模型参数 =====
memory:
  # 向量化基本参数
  embedding:
    enabled: "True"
    provider: "GLM-Embedding"
    dim: "2048"
    batch_size: "100"

  # 精排模型参数（LLM 语义召回）
  llm_re_ranking:
    provider: "GLM-5.2"
    extra_body:
      thinking:
        type: "disabled"

  # 精排模型参数（rerank 模型精排）
  rerank_re_ranking:
    provider: "GLM-5.2"
    extra_body:
      thinking:
        type: "disabled"

# ===== 单轮聊天（记忆提取/整理/进化、意图压缩）=====
simple_chat:
  provider: "GLM-5.2"
  extra_body:
    thinking:
      type: "disabled"
