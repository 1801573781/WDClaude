@chcp 65001 >nul
rem ===== 代理配置（2026-10-09 排查 Connection error 后默认关闭）=====
rem 原代理地址已失效（DNS 解析失败：getaddrinfo failed），且 GLM 为国内直连服务无需代理。
rem 如需恢复：取消下面两行注释并填入有效代理地址。
rem set http_proxy=http://127.0.0.1:7890
rem set https_proxy=http://127.0.0.1:7890
rem set NODE_TLS_REJECT_UNAUTHORIZED=0

rem Phase 1 重构后：包位于 src/wdclaude（src 布局），开发态以 PYTHONPATH 运行
set PYTHONPATH=%~dp0src
python -m wdclaude


