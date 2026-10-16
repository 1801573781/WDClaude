"""首启 bootstrap（v2.0 第 5 章，幂等）。

B/.wdclaude/config/config.yaml 不存在时：创建全套目录骨架 + 拷贝包内 defaults
（config.yaml、model_key.yaml 模板、role/、memory/）+ 依赖自检 + 密钥引导。
"""
import importlib.util
import shutil

from wdclaude.paths import paths

_OPTIONAL_GROUPS = {
    'memory': ['numpy', 'faiss', 'openpyxl'],
    'search': ['requests', 'httpx'],
}


def ensure_first_run(interactive: bool = True) -> None:
    """首启初始化（幂等）。已初始化的项目直接跳过。"""
    first_run = not paths.b_config.exists()
    paths.ensure_data_dirs()

    # 密钥模板自愈（独立于首启判定）：model_key.yaml 被误删时重启自动补拷
    # 模板；已填写的真实密钥文件不受影响（仅在文件缺失时才复制）。
    tpl = paths.pkg_defaults / 'model_key.yaml.tpl'
    if tpl.exists() and not paths.b_model_key.exists():
        shutil.copy2(tpl, paths.b_model_key)
        print(f"[WDClaude] 已生成密钥模板 {paths.b_model_key}")
        print("[WDClaude] 请编辑填写 API Key（将 *** 替换为真实密钥），"
              "或设置环境变量 WDCLAUDE_API_KEY")

    if not first_run:
        return

    print(f"[WDClaude] 首次在此项目运行：初始化 {paths.data_root}")

    # 出厂默认 config.yaml → B/.wdclaude/config/
    default_cfg = paths.pkg_defaults / 'config.yaml'
    if default_cfg.exists() and not paths.b_config.exists():
        shutil.copy2(default_cfg, paths.b_config)
        print(f"[WDClaude] 已生成默认配置 {paths.b_config}")

    # 角色提示词与记忆后端配置
    src_role = paths.pkg_defaults / 'roles'
    dst_role = paths.b_config_dir / 'role'
    if src_role.is_dir():
        shutil.copytree(src_role, dst_role, dirs_exist_ok=True)
        print(f"[WDClaude] 已拷贝默认角色提示词 -> {dst_role}")

    src_mem = paths.pkg_defaults / 'memory'
    dst_mem = paths.b_config_dir / 'memory'
    if src_mem.is_dir():
        shutil.copytree(src_mem, dst_mem, dirs_exist_ok=True)
        print(f"[WDClaude] 已拷贝记忆后端配置 -> {dst_mem}")

    if interactive:
        check_optional_deps()


def check_optional_deps() -> None:
    """可选依赖组自检（缺失时打印安装指引，不中断启动）。"""
    for group, modules in _OPTIONAL_GROUPS.items():
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        if missing:
            print(f"[WDClaude] 检测到未安装可选依赖组 [{group}]（缺失: {', '.join(missing)}）")
            print(f"[WDClaude] 相关功能将自动降级。如需完整功能：pip install wdclaude[{group}]")
