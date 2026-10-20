"""WDClaude 路径中枢（v2.0 启动即项目，设计文档 wdclaude_v2_launch_data_model.md）。

两层模型：
- A1 包代码层：site-packages/wdclaude 或开发仓库 src/wdclaude（含 defaults/ 出厂默认）
- B  项目目录 = 启动时的 cwd（唯一，无参数可覆盖）

核心原则：wdclaude 在哪里启动，哪里就是 B；全部运行数据收纳于 B/.wdclaude/。
"""
from pathlib import Path

# 仅用于测试沙箱（doctor/bootstrap 隔离验证），普通用户无感知
ENV_SANDBOX_B = 'WDCLAUDE_SANDBOX_B'


def escape_project_root(project_root) -> str:
    """将 project_root 完整路径转义为目录名：冒号、正斜杠、反斜杠统一替换为下划线。

    示例：D:\\AI\\WDClaude → D__AI_WDClaude
    """
    s = str(project_root)
    return s.replace(':', '_').replace('\\', '_').replace('/', '_')


class PathResolver:
    """唯一路径解析器（v2.0 两层模型，无 A2 数据层）。"""

    # ── A1：包代码层 ──
    @property
    def pkg_root(self) -> Path:
        """A1 包代码根，与安装位置无关。"""
        return Path(__file__).resolve().parent

    @property
    def pkg_defaults(self) -> Path:
        """A1 包内 defaults/ 出厂默认资源。"""
        return self.pkg_root / 'defaults'

    @property
    def pkg_skills(self) -> Path:
        """A1 包内 skills/ 内置技能。"""
        return self.pkg_root / 'skills'

    # ── B：项目目录 = cwd ──
    @property
    def b_root(self) -> Path:
        """B 项目目录 = 启动时 cwd（唯一锚点，无覆盖参数）。

        WDCLAUDE_SANDBOX_B 仅服务于自动化测试沙箱，普通用户无感知。
        """
        import os
        env = os.environ.get(ENV_SANDBOX_B)
        if env:
            return Path(env).expanduser().resolve()
        return Path.cwd()

    @property
    def data_root(self) -> Path:
        """B/.wdclaude —— wdclaude 全部运行数据的根。"""
        return self.b_root / '.wdclaude'

    @property
    def b_config_dir(self) -> Path:
        """B/.wdclaude/config（config.yaml、model_key.yaml、role/、memory/）。"""
        return self.data_root / 'config'

    @property
    def b_config(self) -> Path:
        """B/.wdclaude/config/config.yaml。"""
        return self.b_config_dir / 'config.yaml'

    @property
    def b_model_key(self) -> Path:
        """B/.wdclaude/config/model_key.yaml（密钥红线，不入 git）。"""
        return self.b_config_dir / 'model_key.yaml'

    @property
    def b_spec_dir(self) -> Path:
        """B/spec（项目资产，留 B 根，不入 .wdclaude）。"""
        return self.b_root / 'spec'

    @property
    def b_tree_cache(self) -> Path:
        """B 的目录树缓存（/init 生成）。"""
        return self.b_root / '.tree_cache.md'

    @property
    def b_wdclaude_md(self) -> Path:
        """B/.wdclaude/wdclaude.md 项目上下文（可选）。"""
        return self.data_root / 'wdclaude.md'

    # ── /init 输出目录 ──
    @property
    def init_root(self) -> Path:
        """B/.wdclaude/init（/init 输出三件套的根目录）。"""
        return self.data_root / 'init'

    def init_dir_for(self, project_root) -> Path:
        """B/.wdclaude/init/转义后的 project_root（/init 输出目录，由调用方按需创建）。"""
        return self.init_root / escape_project_root(project_root)

    # ── .wdclaude 下各数据目录 ──
    @property
    def code_output_dir(self) -> Path:
        """临时代码输出目录 B/.wdclaude/tmp_output（探索/测试/草稿代码）。"""
        return self.data_root / 'tmp_output'

    @property
    def logs_dir(self) -> Path:
        """运行日志目录 B/.wdclaude/log。"""
        return self.data_root / 'log'

    @property
    def sessions_dir(self) -> Path:
        """会话日志目录 B/.wdclaude/log/sessions（不再按哈希分目录）。"""
        return self.logs_dir / 'sessions'

    @property
    def memory_root(self) -> Path:
        """记忆库根 B/.wdclaude/memory_storage（单一库，无路由）。"""
        return self.data_root / 'memory_storage'

    @property
    def skills_dir(self) -> Path:
        """用户/项目技能目录 B/.wdclaude/skills。"""
        return self.data_root / 'skills'

    @property
    def stats_dir(self) -> Path:
        """token 统计目录 B/.wdclaude/token_statistics。"""
        return self.data_root / 'token_statistics'

    @property
    def backups_dir(self) -> Path:
        """str_replace 备份目录 B/.wdclaude/backups。"""
        return self.data_root / 'backups'

    @property
    def cache_dir(self) -> Path:
        """web_search 缓存等 B/.wdclaude/cache。"""
        return self.data_root / 'cache'

    @property
    def media_root(self) -> Path:
        """B/.wdclaude/media（多模态素材根：特写产物与远程下载缓存）。"""
        return self.data_root / 'media'

    @property
    def media_download_dir(self) -> Path:
        """B/.wdclaude/media/download（远程图像落地缓存）。"""
        return self.media_root / 'download'

    @property
    def media_view_dir(self) -> Path:
        """B/.wdclaude/media/view（特写裁剪产物）。"""
        return self.media_root / 'view'

    @property
    def commands_dir(self) -> Path:
        """斜杠命令目录 B/.wdclaude/commands（唯一锚点）。"""
        return self.data_root / 'commands'

    # ── 密钥兜底（唯一保留的 A2 遗迹，仅此一个文件）──
    @property
    def home_model_key(self) -> Path:
        """~/.wdclaude/model_key.yaml 密钥第三级兜底。"""
        return Path.home() / '.wdclaude' / 'model_key.yaml'

    # ── 两级资源查找 ──
    def skill_dirs(self) -> list:
        """技能查找：B/.wdclaude/skills → 包内 wdclaude/skills。"""
        return [self.skills_dir, self.pkg_skills]

    def role_dirs(self, role: str) -> list:
        """角色提示词查找：B/.wdclaude/config/role/<role> → 包内 defaults/role/<role>。"""
        return [self.b_config_dir / 'role' / role, self.pkg_defaults / 'roles' / role]

    def memory_cfg_dirs(self) -> list:
        """记忆后端配置查找：B/.wdclaude/config/memory → 包内 defaults/memory。"""
        return [self.b_config_dir / 'memory', self.pkg_defaults / 'memory']

    # ── 统一入口（保留 v1 协议兼容）──
    def resolve(self, kind: str, *args):
        """按 kind 解析路径。"""
        mapping = {
            'pkg_root': self.pkg_root,
            'pkg_defaults': self.pkg_defaults,
            'b_root': self.b_root,
            'data_root': self.data_root,
            'b_config_dir': self.b_config_dir,
            'b_config': self.b_config,
            'b_model_key': self.b_model_key,
            'b_spec_dir': self.b_spec_dir,
            'b_tree_cache': self.b_tree_cache,
            'b_wdclaude_md': self.b_wdclaude_md,
            'code_output_dir': self.code_output_dir,
            'logs_dir': self.logs_dir,
            'sessions_dir': self.sessions_dir,
            'memory_root': self.memory_root,
            'skills_dir': self.skills_dir,
            'stats_dir': self.stats_dir,
            'backups_dir': self.backups_dir,
            'cache_dir': self.cache_dir,
            'media_root': self.media_root,
            'media_download_dir': self.media_download_dir,
            'media_view_dir': self.media_view_dir,
            'commands_dir': self.commands_dir,
            'init_root': self.init_root,
            'home_model_key': self.home_model_key,
        }
        if kind == 'skill_dirs':
            return self.skill_dirs()
        if kind == 'role_dirs':
            return self.role_dirs(args[0])
        if kind == 'memory_cfg_dirs':
            return self.memory_cfg_dirs()
        if kind in mapping:
            return mapping[kind]
        valid = sorted(set(mapping) | {'skill_dirs', 'role_dirs', 'memory_cfg_dirs'})
        raise KeyError(f"未知路径类型: {kind}（可选: {valid}）")

    def ensure_data_dirs(self):
        """首启 bootstrap：创建 .wdclaude 全套目录骨架。"""
        dirs = [
            self.data_root,
            self.b_config_dir,
            self.b_config_dir / 'role',
            self.b_config_dir / 'memory',
            self.code_output_dir,
            self.logs_dir,
            self.sessions_dir,
            self.memory_root,
            self.skills_dir,
            self.stats_dir,
            self.backups_dir,
            self.cache_dir,
            self.media_root,
            self.media_download_dir,
            self.media_view_dir,
            self.data_root / 'context',
            self.data_root / 'data',
            self.data_root / 'docs',
            self.data_root / 'logo',
            self.data_root / 'tests',
            self.commands_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# 模块级单例，全局唯一
paths = PathResolver()
