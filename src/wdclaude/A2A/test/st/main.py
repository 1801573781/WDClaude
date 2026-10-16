"""
SystemTest A2A 服务入口

提供系统测试执行器的 REST API，支持系统测试用例的执行。
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from python_a2a import A2AServer

from .agent_card import get_agent_card
from ..models import (
    RunSystemTestRequest, RunSystemTestResponse,
    TestRunState, TestStatus,
)
from .system_test_runner import SystemTestRunner
from ..sandbox import SandboxManager
from ..judge import LLMJudge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 应用初始化
# ---------------------------------------------------------------------------

app = FastAPI(title="SystemTest Agent", version="1.0.0")
sandbox_mgr = SandboxManager()
judge = LLMJudge()

# A2A Server 包装
a2a_server = A2AServer(app=app, agent_card=get_agent_card())


@app.get("/.well-known/agent-card.json")
async def serve_agent_card():
    """标准 A2A Agent Card 发现端点"""
    return JSONResponse(content=jsonable_encoder(get_agent_card()))


# ------------------------------ 健康检查 ---------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "test",
        "docker_available": sandbox_mgr.is_available(),
    }


# ------------------------------ 系统测试 --------------------------------

@app.post("/a2a/run_system_tests", response_model=RunSystemTestResponse)
async def run_system_tests(req: RunSystemTestRequest):
    """执行系统测试用例（在沙箱中启动 MyClaude，LLM 评判结果）"""
    t0 = time.perf_counter()
    task_id = req.task_id or f"st-{int(t0)}"

    logger.info("Starting system-test run [task_id=%s] cases=%d",
                task_id, len(req.test_cases))

    runner = SystemTestRunner(sandbox_mgr=sandbox_mgr, judge=judge)
    results = runner.execute(
        test_cases=req.test_cases,
        myclaude_root=req.myclaude_root,
    )

    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    total = len(results)
    elapsed = round(time.perf_counter() - t0, 2)

    logger.info("System tests complete [task_id=%s] %d/%d passed (%.1f%%) in %.1fs",
                task_id, passed, total, passed / total * 100 if total else 0, elapsed)

    # 生成 Excel 报告
    report_path = None
    try:
        report_output_dir = getattr(req, "report_output_dir", None)
        report_path = SystemTestRunner.generate_excel_report(
            results,
            output_dir=report_output_dir,
        )
    except Exception as report_err:
        logger.error("Failed to generate system-test Excel report: %s", report_err)

    return RunSystemTestResponse(
        task_id=task_id,
        state=TestRunState.COMPLETED,
        passed=passed,
        total=total,
        pass_rate=passed / total if total else 0.0,
        details=results,
        execution_time_seconds=elapsed,
        report_path=str(report_path) if report_path else None,
    )


# ------------------------------ 指标端点 ---------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点（简化版）"""
    return JSONResponse(content={
        "service": "test",
        "uptime_seconds": time.perf_counter(),
        "docker_available": sandbox_mgr.is_available(),
    })


# ========================================================================
# CLI 模式：python -m wdclaude.A2A.test.st.main --json D:/.../u9.json
# ========================================================================

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from wdclaude.utility.config_loader import global_cfg

    parser = argparse.ArgumentParser(
        description="SystemTest CLI — 执行系统测试用例"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="测试用例 JSON 文件路径",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="日志文件全路径（可选，默认输出到 logs_root/systemtest_cli.log）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出目录（默认使用 config 中 logs_root）",
    )
    parser.add_argument(
        "--myclaude-root",
        default=None,
        help="MyClaude 源码根目录（默认使用 config 中 project_root）",
    )
    args = parser.parse_args()

    # ── 配置日志 ──────────────────────────────────────────────────
    logs_root = Path(args.output) if args.output else Path(global_cfg.base_path.logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        log_file = Path(args.log)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    else:
        handlers.append(logging.FileHandler(
            logs_root / "systemtest_cli.log", encoding="utf-8",
        ))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    myclaude_root = args.myclaude_root or global_cfg.base_path.project_root
    judge = LLMJudge()

    # ── 加载测试用例 ──────────────────────────────────────────────
    json_path = Path(args.json)
    if not json_path.exists():
        logger.error("JSON 文件不存在: %s", json_path)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    logger.info("从 %s 加载了 %d 条系统测试用例", json_path, len(test_cases))

    # ── 执行系统测试 ──────────────────────────────────────────────
    runner = SystemTestRunner(sandbox_mgr=sandbox_mgr, judge=judge)
    results = runner.execute(
        test_cases=test_cases,
        myclaude_root=myclaude_root,
    )

    report_path = SystemTestRunner.generate_excel_report(
        results,
        output_dir=str(logs_root),
    )

    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    total = len(results)
    failed = total - passed
    print("\n" + "=" * 60)
    print("  系统测试完成")
    print(f"  通过: {passed}  失败: {failed}  合计: {total}")
    print(f"  通过率: {passed / total * 100:.1f}%" if total else "  无测试用例")
    print(f"  Excel 报告: {report_path}")
    print("=" * 60)
    sys.exit(0)
