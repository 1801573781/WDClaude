import sys
import logging

# import asyncio

# ===== 日志文件配置（必须在其他 import 之前，避免日志落到 stderr）=====
# Phase 2（设计文档 P3）：运行日志改走 A2（~/.wdclaude/logs），安装后不再写代码目录
from wdclaude.paths import paths as _paths

_log_dir = _paths.logs_dir
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            _log_dir / "wdclaude.log",
            encoding="utf-8"
        )
    ],
)


import time
import threading

# MyClaudeCLI 的导入延迟到 main() 内部，以便在导入前显示启动动画


def _start_spinner(message="正在启动中，请等待几秒..."):
    """启动一个 spinner 线程，返回 (停止事件, 线程对象, 输出锁)。"""
    is_tty = sys.stdout.isatty()
    spinner_chars = ('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')
    all_done = threading.Event()
    output_lock = threading.Lock()
    start_time = time.time()

    def _spin():
        i = 0
        while not all_done.is_set():
            char = spinner_chars[i % len(spinner_chars)]
            with output_lock:
                if all_done.is_set():
                    break
                if is_tty:
                    elapsed = int(time.time() - start_time)
                    msg = f"  {char} {message} ({elapsed}s)"
                    sys.stdout.write(f"\r{msg.ljust(70)}")
                    sys.stdout.flush()
            time.sleep(0.15)
            i += 1

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return all_done, thread, output_lock


def _stop_spinner(all_done, thread, output_lock):
    """停止 spinner 线程并清除行。"""
    all_done.set()
    thread.join(timeout=1.0)
    if sys.stdout.isatty():
        with output_lock:
            sys.stdout.write(f"\r{' ' * 70}\r")
            sys.stdout.flush()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="WDClaude Code CLI")

    parser.add_argument(
        '-r', '--role',
        type=str,
        default='mycode',
        help='角色名称，决定加载哪套提示词（默认：mycode）'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        default=False,
        help='进入测试模式（必须与 --prompt 成对使用）'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='测试模式下的用户输入（必须与 --test-mode 成对使用）'
    )
    parser.add_argument(
        '--test-output',
        type=str,
        default=None,
        help='测试模式下输出结构化JSON结果的文件路径（仅与 --test-mode 一起使用）'
    )
    parser.add_argument(
        '--version',
        action='store_true',
        default=False,
        help='显示版本号并退出'
    )
    parser.add_argument(
        'command',
        nargs='?',
        default=None,
        help='子命令：doctor（环境体检）'
    )

    args = parser.parse_args()

    # 验证 --test-mode 与 --prompt 必须成对出现
    if args.test_mode and args.prompt is None:
        print("错误：--test-mode 必须与 --prompt 成对使用。")
        print("用法：WDClaude --test-mode --prompt \"your prompt here\"")
        sys.exit(1)
    if args.prompt is not None and not args.test_mode:
        print("错误：--prompt 必须与 --test-mode 成对使用。")
        print("用法：WDClaude --test-mode --prompt \"your prompt here\"")
        sys.exit(1)

    if args.role != 'mycode':
        print("暂时不支持此类角色，程序退出。")
        sys.exit(1)

    # ── 版本号 ──
    if args.version:
        from wdclaude import __version__
        print(f"wdclaude {__version__}")
        return

    # ── 子命令：doctor（不进入交互循环）──
    if args.command == 'doctor':
        from wdclaude.doctor import run_doctor
        sys.exit(run_doctor())
    if args.command is not None:
        print(f"未知子命令：{args.command}（可用：doctor）")
        sys.exit(1)

    # v2.0（启动即项目）：无场景检测、无 B 覆盖参数——cwd 即 B，
    # 修改目标由用户对话指令中的路径决定。

    # 测试模式：跳过欢迎页面和 spinner，直接初始化（不注册交互信号层，
    # 保证测试子进程的信号语义与 CI 环境一致，Ctrl+C 直接终止）
    if args.test_mode:
        # 目录骨架 + 配置自愈（非交互、幂等）：补齐沙箱子进程的写路径
        # （sessions / memory_storage / token_statistics 等）
        from wdclaude.bootstrap import ensure_first_run
        ensure_first_run(interactive=False)
        from wdclaude.cli.mycli import MyClaudeCLI
        cli = MyClaudeCLI(role=args.role)
        cli.run_test_mode(args.prompt, test_output_path=args.test_output)
        return

    # 首启 bootstrap（v2.0 第 5 章）：B/.wdclaude 骨架 + 默认配置 + 依赖自检
    from wdclaude.bootstrap import ensure_first_run
    ensure_first_run(interactive=True)

    print()
    spinner_done, spinner_thread, spinner_lock = _start_spinner()

    try:
        from wdclaude.cli.mycli import MyClaudeCLI
        cli = MyClaudeCLI(role=args.role)
    finally:
        _stop_spinner(spinner_done, spinner_thread, spinner_lock)

    print()
    print("  ✅ 启动完毕，欢迎使用 WDClaude")
    print()

    # ===== 优雅退出：注册交互模式信号层（原生控制台处理器 +
    # Python 信号处理器 + atexit 兜底），此后 Ctrl+C 走两级中断 =====
    from wdclaude.utility.shutdown import shutdown_mgr, install_signal_handlers
    install_signal_handlers()

    # CLI 主循环返回后透传最终退出码（正常 0 / 中断 130 / 异常 1）
    try:
        cli.run()
        exit_code = shutdown_mgr.final_exit_code
    except KeyboardInterrupt:
        # 兜底：主循环外残留的中断（理论不可达，防御性捕获）
        shutdown_mgr.run_shutdown(reason="interrupt")
        exit_code = shutdown_mgr.final_exit_code
    except Exception:
        import traceback
        traceback.print_exc()
        shutdown_mgr.run_shutdown(reason="error", exit_code=1)
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
