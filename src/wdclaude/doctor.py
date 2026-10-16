"""wdclaude doctor —— 环境体检（v2.0 第 3/9.4 章）。

检查项锚定 B/.wdclaude：写权限、config/model_key 存在性、包根与 B 路径、
核心与可选依赖。报障时的标准信息收集手段。
"""
import importlib.util
import os
import sys

from wdclaude.paths import paths


def _check() -> list:
    results = []

    results.append((f"Python >= 3.10（当前 {sys.version.split()[0]}）",
                    sys.version_info >= (3, 10), ''))

    for m in ('openai', 'rich', 'yaml'):
        results.append((f"核心依赖 {m}", importlib.util.find_spec(m) is not None, ''))

    for group, mods in {'memory': ['numpy', 'faiss', 'openpyxl'],
                        'search': ['requests', 'httpx']}.items():
        for m in mods:
            results.append((f"可选依赖 [{group}] {m}",
                            importlib.util.find_spec(m) is not None, ''))

    # B/.wdclaude 写权限
    try:
        paths.ensure_data_dirs()
        probe = paths.data_root / '.write_probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        results.append(('B/.wdclaude 写权限', True, str(paths.data_root)))
    except Exception as exc:  # noqa: BLE001
        results.append(('B/.wdclaude 写权限', False, str(exc)))

    # 配置与密钥（密钥三级：B 级文件 / 兜底文件 / 环境变量）
    results.append(('B/.wdclaude/config/config.yaml', paths.b_config.exists(),
                    str(paths.b_config)))
    has_key = (paths.b_model_key.exists()
               or paths.home_model_key.exists()
               or bool(os.environ.get('WDCLAUDE_API_KEY')))
    results.append(('API 密钥（B 级 model_key.yaml / ~/.wdclaude 兜底 / 环境变量）',
                    has_key, ''))

    # 路径展示
    results.append(('包根 A1', True, str(paths.pkg_root)))
    results.append(('项目 B', True, str(paths.b_root)))
    results.append(('数据根 B/.wdclaude', True, str(paths.data_root)))
    results.append(('会话目录', True, str(paths.sessions_dir)))
    return results


def run_doctor() -> int:
    """执行体检并打印报告；全部通过返回 0，存在失败项返回 1。"""
    results = _check()
    width = max(len(name) for name, _, _ in results)
    failed = 0
    print()
    print("  WDClaude doctor —— 环境体检报告")
    print("  " + "=" * (width + 14))
    for name, ok, detail in results:
        mark = 'PASS' if ok else 'FAIL'
        if not ok:
            failed += 1
        line = f"  [{mark}] {name.ljust(width)}"
        if detail:
            line += f"  {detail}"
        print(line)
    print("  " + "=" * (width + 14))
    print(f"  结论: {'全部通过' if failed == 0 else f'{failed} 项失败，详见上方 FAIL 行'}")
    print()
    return 0 if failed == 0 else 1
