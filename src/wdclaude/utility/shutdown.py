"""优雅退出协调器（ShutdownCoordinator）。

架构角色：全进程生命周期的"单一关闭出口"（优雅退出方案核心模块）。

核心机制：
1. 单一出口 —— /quit、Ctrl+C（空闲/执行中）、Ctrl+Break、未捕获异常等
   所有退出路径，最终汇聚到 run_shutdown()，按优先级执行清理链后退出。
2. 两级中断 —— 第一次 Ctrl+C 置位中断标志（安全点优雅退出）；第二次
   Ctrl+C 硬退出（TerminateProcess，放弃清理，atexit 不执行）。
3. 安全点轮询 —— QueryLoop（Turn 边界/工具执行前）、chat_llm（流式
   chunk 间隙）、CLI 主循环、A2A 测试用例循环通过 is_interrupted()
   轮询标志，置位后在安全点有序退出，而非在任意代码行中断。
4. 清理动作分级 —— 必须完成（会话日志落盘，由 QueryLoop 自身的
   run()/finalize_interrupted() 保证）＞ 尽力完成（屏幕缓冲保存、
   中断记忆补存）＞ 可跳过（A2A 服务处置）；含 LLM 调用的耗时操作
   （记忆整理/进化）在中断场景一律跳过。

清理链优先级（数值大者先执行）：
  100 terminal_restore   终端状态还原（补换行，避免 Rich 残留行污染摘要）
   90 （保留）SessionLog 落盘 —— 由 QueryLoop.run()/finalize_interrupted() 保证
   80 screen_buffer_save  屏幕输出缓冲保存为 HTML（mycli 注册）
   60 a2a_services        A2A 服务子进程处置（按 kill_a2a_on_exit 配置）
   10 logging.shutdown    由 run_shutdown() 内置最后执行

依赖纪律：本模块不依赖 cli_print / query_loop / mycli（保持单向依赖，
避免清理链触发循环导入）；config_loader 惰性导入，容忍配置缺失。
"""
import atexit
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 退出码约定
EXIT_OK = 0           # 正常退出（/quit、EOF）
EXIT_INTERRUPT = 130  # 用户中断（SIGINT 惯例退出码）
EXIT_ERROR = 1        # 异常退出


def get_shutdown_cfg() -> SimpleNamespace:
    """安全读取 shutdown 配置节（缺失项回退默认值）。

    兼容旧配置文件（无 shutdown 节）与异常环境（配置未加载）。
    """
    defaults = {
        "confirm_on_ctrl_c": True,          # 空闲时 Ctrl+C 触发的退出是否二次确认
        "save_session_html_on_exit": True,  # 退出时自动保存屏幕输出 HTML
        "kill_a2a_on_exit": "never",        # A2A 服务子进程处置: never / always / ask
        "cleanup_timeout_seconds": 10.0,    # 清理链总超时（秒），超时放弃剩余清理
    }
    try:
        from wdclaude.utility.config_loader import global_cfg
        cfg = getattr(global_cfg, "shutdown", None)
        if cfg is not None:
            for key in defaults:
                value = getattr(cfg, key, None)
                if value is not None:
                    defaults[key] = value
    except Exception:
        pass
    return SimpleNamespace(**defaults)


def _write_stderr(msg: str):
    """低层 stderr 写入：退出路径上 Rich/日志系统可能已不可用。"""
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass


def _hard_exit(code: int):
    """硬退出：TerminateProcess 立即终止进程。

    用于第二次 Ctrl+C 的强制退出路径——用户已明确表达"等不了"，
    放弃全部清理。TerminateProcess 绕过 ExitProcess 的 DLL 卸载通知
    （其他线程持锁时可能死锁），是任意阻塞状态下最可靠的退出手段。
    atexit 在此路径下不会执行（符合预期）。
    """
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    except Exception:
        pass
    os._exit(code)


class _CleanupTask:
    """清理任务条目：名称 + 回调 + 优先级（数值大者先执行）。"""

    __slots__ = ("name", "callback", "priority")

    def __init__(self, name: str, callback: Callable[[], None], priority: int):
        self.name = name
        self.callback = callback
        self.priority = priority


class ShutdownCoordinator:
    """优雅退出协调器（模块级单例 shutdown_mgr）。

    设计契约：
    1. 单一出口：所有退出路径汇聚到 run_shutdown()（幂等）。
    2. 两级中断：首次 Ctrl+C 置位标志优雅退出；再次 Ctrl+C 硬退出。
    3. 安全点轮询：各模块通过 is_interrupted() 在安全点有序退出。
    4. 清理分级：日志落盘（QueryLoop 自保证）＞ 屏幕保存/记忆补存
       （尽力）＞ A2A 处置（可跳过）；清理链单步异常隔离，总超时兜底。
    """

    def __init__(self):
        # ── 中断状态（线程安全 Event，任意线程可置位/轮询） ──
        self._interrupt_event = threading.Event()
        self._interrupt_reason = ""

        # ── 关闭流程状态（幂等保护） ──
        self._shutdown_started = threading.Event()
        self._lock = threading.Lock()
        self._reason_override: Optional[str] = None
        self._exit_code = EXIT_OK
        self._reason = "normal"

        # ── 清理任务注册表 ──
        self._cleanup_tasks: List[_CleanupTask] = []

        # ── A2A 服务子进程登记表 [(Popen, name), ...] ──
        self._a2a_processes: List[Tuple] = []

        # ── 退出摘要数据 ──
        self._stats_provider: Optional[Callable[[], dict]] = None
        self._artifacts: List[Tuple[str, str]] = []  # (label, path)
        self._start_time: Optional[datetime] = None

    # ══════════════════ 中断标志管理 ══════════════════

    def request_interrupt(self, reason: str = "user", quiet: bool = False):
        """请求中断：置位全局中断标志（幂等，不打印任何提示）。

        由信号层（原生处理器/Python 信号处理器）调用；各安全点通过
        is_interrupted() 轮询此标志后有序退出，退出确认由 CLI 层在安全点
        以一行式提示统一弹出（"中断请求已收到。确认退出…？（Y/N）"），
        避免 Ctrl+C 瞬间出现多行啰嗦提示。quiet 参数仅为兼容旧调用保留。
        """
        if self._interrupt_event.is_set():
            return
        self._interrupt_event.set()
        self._interrupt_reason = reason

    def mark_interrupt(self, reason: str = "user"):
        """静默置位中断标志（内层 KeyboardInterrupt 捕获点调用）。

        信号层不打印提示，此处仅确保标志可靠置位；退出确认由 CLI 层
        在安全点统一弹出。
        """
        self.request_interrupt(reason, quiet=True)

    def clear_interrupt(self):
        """清除中断标志（退出确认被用户取消时调用，恢复会话）。"""
        self._interrupt_event.clear()
        self._interrupt_reason = ""

    def is_interrupted(self) -> bool:
        """安全点轮询接口：中断标志是否已置位。"""
        return self._interrupt_event.is_set()

    @property
    def interrupt_reason(self) -> str:
        return self._interrupt_reason

    # ══════════════════ 注册接口 ══════════════════

    def register_cleanup(self, name: str, callback: Callable[[], None], priority: int = 50):
        """注册清理任务（同名重复注册时替换）。

        Args:
            name: 任务名（退出摘要中展示）
            callback: 无参回调，异常由 run_shutdown() 隔离捕获
            priority: 优先级，数值大者先执行（100 终端还原 → 10 日志收尾）
        """
        with self._lock:
            self._cleanup_tasks = [t for t in self._cleanup_tasks if t.name != name]
            self._cleanup_tasks.append(_CleanupTask(name, callback, priority))

    def set_stats_provider(self, provider: Callable[[], dict]):
        """注册会话统计提供者（退出摘要卡片数据源）。

        provider 返回 dict，可含：queries(int)、tokens(dict)、
        logs(list[(label, path)])。
        """
        self._stats_provider = provider

    def note_artifact(self, label: str, path: str):
        """登记退出产出物路径（清理任务保存文件后调用，摘要中展示）。"""
        self._artifacts.append((label, str(path)))

    def register_a2a_process(self, proc, name: str):
        """登记 A2A 服务子进程（退出时按 kill_a2a_on_exit 配置处置）。"""
        with self._lock:
            # 顺带剔除已退出的句柄
            self._a2a_processes = [
                (p, n) for p, n in self._a2a_processes if p.poll() is None
            ]
            if proc.poll() is None:
                self._a2a_processes.append((proc, name))

    def set_reason(self, reason: str):
        """显式设置退出原因（异常路径调用，如 "error"）。"""
        self._reason_override = reason

    # ══════════════════ 统一关闭入口 ══════════════════

    @property
    def final_exit_code(self) -> int:
        """run_shutdown() 解析出的最终退出码（供 __main__ 的 sys.exit 使用）。"""
        return self._exit_code

    def run_shutdown(self, reason: str = None, exit_code: int = None):
        """统一关闭入口：按优先级执行清理链，打印退出摘要。幂等。

        Args:
            reason: 退出原因（normal/interrupt/atexit/error）；
                    None 时优先取 set_reason() 的显式值，再按中断标志推导
            exit_code: 显式退出码；None 时按中断标志推导（130/0）
        """
        if self._shutdown_started.is_set():
            return
        self._shutdown_started.set()

        # 推导退出原因
        if reason is None:
            reason = self._reason_override or (
                "interrupt" if self._interrupt_event.is_set() else "normal"
            )
        self._reason = reason
        # 推导退出码
        if exit_code is None:
            exit_code = EXIT_INTERRUPT if self._interrupt_event.is_set() else EXIT_OK
        self._exit_code = exit_code

        cfg = get_shutdown_cfg()
        deadline = time.monotonic() + float(cfg.cleanup_timeout_seconds)

        # 按优先级降序执行清理链（单步异常隔离，失败不阻断后续）
        with self._lock:
            tasks = sorted(self._cleanup_tasks, key=lambda t: -t.priority)
        results = []
        for task in tasks:
            if time.monotonic() >= deadline:
                results.append((task.name, False, "清理总超时，跳过"))
                continue
            try:
                task.callback()
                results.append((task.name, True, ""))
            except KeyboardInterrupt:
                # 清理期间再按 Ctrl+C：结束剩余清理（硬退出由原生层兜底）
                results.append((task.name, False, "被键盘中断"))
                break
            except Exception as e:
                results.append((task.name, False, str(e)))

        # 打印退出摘要
        try:
            self._print_summary(results)
        except Exception:
            pass

        # 日志系统收尾（此后 logger 输出不再可靠）
        try:
            logging.shutdown()
        except Exception:
            pass

    # ══════════════════ 内置清理任务 ══════════════════

    @staticmethod
    def builtin_terminal_restore():
        """终端状态还原（优先级 100）：补换行，避免 Rich 残留行污染摘要。"""
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

    def cleanup_a2a_services(self):
        """A2A 服务子进程处置（优先级 60，按 kill_a2a_on_exit 配置）。

        - never（默认）：保留运行（服务为跨会话复用而设计）
        - always：退出时 terminate 全部
        - ask：交互询问
        """
        cfg = get_shutdown_cfg()
        with self._lock:
            procs = [(p, n) for p, n in self._a2a_processes if p.poll() is None]
        if not procs:
            return
        mode = str(cfg.kill_a2a_on_exit).lower()
        if mode == "never":
            logger.info("A2A 服务保留运行: %s", ", ".join(n for _, n in procs))
            return
        if mode == "ask":
            try:
                ans = input(
                    f"\n检测到 {len(procs)} 个 A2A 服务进程"
                    f"（{', '.join(n for _, n in procs)}），退出时是否终止？(y/n): "
                ).strip().lower()
            except Exception:
                ans = "n"
            if ans != "y":
                return
        for proc, name in procs:
            try:
                proc.terminate()
                logger.info("A2A 服务已终止: %s (pid=%s)", name, proc.pid)
            except Exception as e:
                logger.warning("A2A 服务终止失败: %s: %s", name, e)

    # ══════════════════ 退出摘要 ══════════════════

    def _print_summary(self, task_results):
        """打印退出摘要卡片（纯 print，不依赖 Rich——退出路径上最稳妥）。"""
        reason_labels = {
            "normal": "正常退出",
            "interrupt": "用户中断（Ctrl+C）",
            "atexit": "进程收尾",
            "error": "异常退出",
        }
        lines = ["", "═" * 60, "  WDClaude 退出摘要", "═" * 60]
        label = reason_labels.get(self._reason, self._reason)
        lines.append(f"  退出原因: {label} | 退出码: {self._exit_code}")

        if self._start_time is not None:
            total = int((datetime.now() - self._start_time).total_seconds())
            lines.append(
                f"  会话时长: {total // 3600}时{(total % 3600) // 60}分{total % 60}秒"
            )

        stats = {}
        if self._stats_provider is not None:
            try:
                stats = self._stats_provider() or {}
            except Exception:
                stats = {}
        if stats.get("queries") is not None:
            lines.append(f"  Query 数: {stats['queries']}")
        tokens = stats.get("tokens")
        if tokens:
            lines.append(
                f"  Token: 输入(缓存)={tokens.get('prompt_cache_hit', 0):,}, "
                f"输入(未命中)={tokens.get('prompt_cache_miss', 0):,}, "
                f"输出={tokens.get('completion_tokens', 0):,}"
            )

        # 会话日志路径（stats provider 提供）
        for entry in stats.get("logs", []):
            try:
                lbl, path = entry
                lines.append(f"  {lbl}: {path}")
            except Exception:
                continue
        # 清理任务登记的产出物
        for lbl, path in self._artifacts:
            lines.append(f"  {lbl}: {path}")

        failed = [(n, e) for n, ok, e in task_results if not ok]
        if failed:
            lines.append("  清理异常:")
            for n, e in failed:
                lines.append(f"    ✗ {n}: {e}")
        elif task_results:
            lines.append(f"  清理链: {len(task_results)} 项全部完成")
        lines.append("═" * 60)
        # 结尾补一个空行，与终端提示符分隔
        print("\n".join(lines) + "\n")


# ══════════════════ 信号处理 ══════════════════

# 模块级单例
shutdown_mgr = ShutdownCoordinator()

# 原生控制台处理器是否已激活（Windows）
_native_active = False
# ctypes 回调必须保持模块级引用，防止被 GC 后崩溃
_native_handler_ref = None


def _py_signal_handler(signum, frame):
    """Python 层信号处理器：确保标志置位并抛出 KeyboardInterrupt。

    - 原生处理器已激活（Windows）：标志由原生层（OS 线程）置位（不打印
      提示），此处仅抛出 KeyboardInterrupt 中断 Python 层长循环（记忆提取等）
    - 原生层未激活（非 Windows / 安装失败）：本函数承担两级中断逻辑，
      第二次 Ctrl+C（标志已置位）直接硬退出
    """
    if not _native_active:
        if shutdown_mgr.is_interrupted():
            _hard_exit(EXIT_INTERRUPT)
        shutdown_mgr.request_interrupt("Ctrl+C")
    raise KeyboardInterrupt


def _install_native_ctrl_handler() -> bool:
    """Windows：安装原生控制台控制处理器。

    signal.signal 的 Python 回调仅在主线程字节码边界执行——主线程阻塞在
    socket 读取（LLM 流式响应）时无法及时响应。原生处理器运行在 OS 独立
    线程，保证任意阻塞状态下：首次 Ctrl+C 立即置位中断标志（不打印提示，
    确认由 CLI 层在安全点一行式弹出），第二次 Ctrl+C 立即硬退出。

    Returns:
        True 安装成功；False 平台不支持或安装失败（回退 signal.signal 层）
    """
    global _native_active, _native_handler_ref
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        # CTRL_C_EVENT=0 / CTRL_BREAK_EVENT=1；其余（关闭/注销事件）不拦截
        def _handler(event_type):
            try:
                if event_type in (0, 1):
                    if shutdown_mgr.is_interrupted():
                        # 第二次 Ctrl+C：硬退出（任意阻塞状态立即生效）
                        _hard_exit(EXIT_INTERRUPT)
                    shutdown_mgr.request_interrupt("Ctrl+C")
                    # 返回 0（FALSE）：继续传递给 CPython 处理链，
                    # 由 Python 层信号处理器抛出 KeyboardInterrupt
                    return 0
            except Exception:
                pass
            return 0

        _native_handler_ref = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(_handler)
        ok = ctypes.windll.kernel32.SetConsoleCtrlHandler(_native_handler_ref, 1)
        if ok:
            _native_active = True
            logger.info("Windows 原生控制台处理器已安装（两级中断）")
            return True
        logger.warning("SetConsoleCtrlHandler 安装失败，回退 signal.signal 层")
        return False
    except Exception as e:
        logger.warning(f"原生控制台处理器安装异常，回退 signal.signal 层: {e}")
        return False


def install_signal_handlers():
    """注册优雅退出的全部信号层设施（交互模式启动时调用一次）。

    - Windows 原生控制台处理器：两级中断的硬退出保障
    - signal.signal 回退层：SIGINT/SIGBREAK/SIGTERM → 标志置位 + KeyboardInterrupt
    - 内置清理任务：terminal_restore(100) / a2a_services(60)
    - atexit 兜底：覆盖漏掉显式 run_shutdown 的退出路径（幂等）
    """
    shutdown_mgr._start_time = datetime.now()

    # 内置清理任务（同名幂等注册）
    shutdown_mgr.register_cleanup(
        "terminal_restore", ShutdownCoordinator.builtin_terminal_restore, priority=100)
    shutdown_mgr.register_cleanup(
        "a2a_services", shutdown_mgr.cleanup_a2a_services, priority=60)

    global _native_active
    _native_active = _install_native_ctrl_handler()

    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None),
                getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _py_signal_handler)
        except (ValueError, OSError):
            pass  # 非主线程或平台不支持时静默跳过

    # atexit 兜底（run_shutdown 幂等，已显式执行过则静默返回）
    atexit.register(lambda: shutdown_mgr.run_shutdown(reason="atexit"))
    logger.info("优雅退出信号层已就绪（native=%s）", _native_active)
