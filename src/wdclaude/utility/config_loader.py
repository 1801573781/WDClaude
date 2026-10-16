"""配置加载器（v2.0 启动即项目，设计文档第 6 章）。

两级配置 + 环境变量（优先级高 → 低）：
1. 环境变量（WDCLAUDE_API_KEY → model.api_key）
2. B/.wdclaude/config/config.yaml + model_key.yaml（唯一活跃配置）
3. 包内 defaults/config.yaml（出厂默认，仅在 B 无对应文件时兜底）

密钥三级查找：env → B/.wdclaude/config/model_key.yaml → ~/.wdclaude/model_key.yaml。
记忆库单一化：base_dir 固定注入 B/.wdclaude/memory_storage/<backend>，无视 YAML 写死值。
base_path 动态层锚点（v2.0 第 6 章）：logs/backups/skill/tmp_output → B/.wdclaude/，
spec → B/spec，project_root → B。
"""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from wdclaude.paths import paths as _paths


def _dict_to_namespace(obj):
    """递归把字典转成 SimpleNamespace，支持点号访问"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_dict_to_namespace(i) for i in obj]
    return obj


def _load_yaml(path) -> dict:
    """安全读取 YAML；不存在或解析失败返回空 dict。"""
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            return raw if isinstance(raw, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _deep_merge(base, override):
    """递归合并两个字典，override 中的值覆盖 base 的同名字段"""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class _DynamicBasePath:
    """base_path 兼容层：属性值由 PathResolver 动态计算，YAML 中的 base_path 段无效。

    v2.0 锚点（设计文档第 6 章）：
    - project_root → B（启动目录）
    - spec_root → B/spec（项目资产，留 B 根）
    - skill_root / code_output_root / logs_root / backups_root → B/.wdclaude/ 下
    """

    @property
    def project_root(self) -> str:
        return str(_paths.b_root)

    @property
    def code_output_root(self) -> str:
        return str(_paths.code_output_dir)

    @property
    def logs_root(self) -> str:
        return str(_paths.logs_dir)

    @property
    def backups_root(self) -> str:
        return str(_paths.backups_dir)

    @property
    def spec_root(self) -> str:
        return str(_paths.b_spec_dir)

    @property
    def skill_root(self) -> str:
        return str(_paths.skills_dir)

    def __repr__(self) -> str:
        return f"<DynamicBasePath b_root={self.project_root} data_root={_paths.data_root}>"


def load_config(config_yaml="config.yaml", model_key_yaml="model_key.yaml"):
    """两级配置合并 + 密钥三级查找，返回点号访问命名空间。

    优先级（高 → 低）：env > B/.wdclaude/config > 包内 defaults。
    另按 memory.backend 挂载后端配置，并强制注入单一记忆库 base_dir。
    """
    # 3) 包内出厂默认
    merged = _load_yaml(_paths.pkg_defaults / config_yaml)

    # 2) B/.wdclaude/config（唯一活跃配置）
    merged = _deep_merge(merged, _load_yaml(_paths.b_model_key))
    merged = _deep_merge(merged, _load_yaml(_paths.b_config))

    # 1) 密钥三级查找之环境变量（最高优先级）
    env_key = os.environ.get("WDCLAUDE_API_KEY")
    if env_key:
        merged.setdefault("model", {})
        merged["model"]["api_key"] = env_key

    # ---------- 密钥兜底：~/.wdclaude/model_key.yaml（第三级，B 级密钥不可用时）----------
    # 兜底条件 = "B 级密钥不可用"：文件缺失，或虽存在但仍是未填写的模板
    # （首启 bootstrap 拷贝的 model_key.yaml.tpl 占位文件，无真实密钥）。
    # 判定方法：B 级文件的顶层节中不含任何真实 api_key（与模板占位一致）。
    def _b_key_usable() -> bool:
        if not _paths.b_model_key.exists():
            return False
        b_raw = _load_yaml(_paths.b_model_key)
        # 模板特征：所有 provider 节的 api_key 均为占位文本或空
        # 占位形态：'在此填写…' / 'sk-xxx…' / 纯星号 '***'（model_key.yaml.tpl 通配符）
        for v in b_raw.values():
            if isinstance(v, dict):
                key = v.get('api_key')
                if (key and '在此填写' not in str(key)
                        and not str(key).startswith('sk-xxx')
                        and str(key).strip('*')):
                    return True
        return False

    if not _b_key_usable() and _paths.home_model_key.exists():
        merged = _deep_merge(merged, _load_yaml(_paths.home_model_key))

    # ---------- 记忆后端配置（两级查找：B/.wdclaude/config/memory → 包内 defaults/memory）----------
    memory_backend = merged.get("memory", {}).get("backend") if isinstance(merged, dict) else None
    if memory_backend and memory_backend not in ("none", None):
        mem_yaml = None
        for cand_dir in _paths.memory_cfg_dirs():
            cand = cand_dir / f"{memory_backend}.yaml"
            if cand.exists():
                mem_yaml = cand
                break
        if mem_yaml is not None:
            memory_raw = _load_yaml(mem_yaml)
            if memory_raw:
                if memory_backend in memory_raw:
                    memory_config = {memory_backend: memory_raw[memory_backend]}
                else:
                    memory_config = {memory_backend: memory_raw}
                # 主配置（B 级）同名优先
                merged = _deep_merge(memory_config, merged)

    # ---------- 记忆库单一化：base_dir 固定注入（v2.0 第 8 章，无视 YAML 写死值）----------
    backend_cfg = merged.get(memory_backend) if memory_backend else None
    if isinstance(backend_cfg, dict):
        storage = backend_cfg.setdefault("storage", {})
        storage["base_dir"] = str(_paths.memory_root / str(memory_backend))

    # ---------- base_path 动态化 ----------
    merged["base_path"] = _DynamicBasePath()

    return _dict_to_namespace(merged)


# 全局单例：首次 import 加载一次，之后所有模块共享（sys.modules 缓存机制）
global_cfg = load_config()
