"""命令扫描器，启动时递归扫描 .wdclaude/commands/ 目录，发现并注册所有 .md 文件为斜杠命令。"""

from pathlib import Path

import yaml

from wdclaude.command import CommandInfo
from wdclaude.command.registry import CommandRegistry
from wdclaude.paths import paths as _paths


class CommandScanner:
    """扫描 B/.wdclaude/commands/ 目录，将 .md 文件注册为斜杠命令。"""

    def __init__(self, project_root: str | Path, cmd_dir: str | Path | None = None):
        self.project_root = Path(project_root)
        # 默认锚点：paths.commands_dir（B/.wdclaude/commands，与首启骨架一致）；
        # 显式传入 cmd_dir 时仍以 project_root/cmd_dir 解析（兼容旧调用）
        if cmd_dir is None:
            self.cmd_dir = _paths.commands_dir
        else:
            self.cmd_dir = Path(cmd_dir) if Path(cmd_dir).is_absolute() else self.project_root / cmd_dir

    def scan(self) -> CommandRegistry:
        """扫描 .wdclaude/commands/ 目录，返回填充好的 CommandRegistry"""
        registry = CommandRegistry()
        if not self.cmd_dir.exists():
            return registry

        for md_file in sorted(self.cmd_dir.rglob("*.md")):
            try:
                command_name = self._path_to_command(md_file)
                info = self._parse_file(md_file, command_name)
                registry.register(info)
            except Exception as e:
                # 文件读取或解析失败时跳过，打印警告
                print(f"[CommandScanner] 警告: 跳过文件 {md_file}: {e}")

        return registry

    def _path_to_command(self, file_path: Path) -> str:
        """将文件路径转换为斜杠命令名

        .wdclaude/commands/opsx/propose.md → /opsx:propose
        .wdclaude/commands/deploy.md → /deploy
        .wdclaude/commands/test/unit/run.md → /test:unit:run
        """
        relative = file_path.relative_to(self.cmd_dir)
        parts = list(relative.parts)
        parts[-1] = parts[-1].removesuffix(".md")
        return "/" + ":".join(parts)

    def _parse_file(self, file_path: Path, command_name: str) -> CommandInfo:
        """解析 .md 文件，提取元数据和正文"""
        content = file_path.read_text(encoding="utf-8")

        description = ""
        category = "general"
        tags: list[str] = []

        # 解析 YAML Front Matter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                front_matter = content[3:end].strip()
                try:
                    meta = yaml.safe_load(front_matter)
                    if meta:
                        description = meta.get("description", "")
                        category = meta.get("category", "general")
                        tags_val = meta.get("tags", [])
                        if isinstance(tags_val, list):
                            tags = tags_val
                        elif isinstance(tags_val, str):
                            tags = [tags_val]
                except yaml.YAMLError:
                    # YAML 解析失败，使用默认值
                    pass
                content = content[end + 3:].strip()

        return CommandInfo(
            command_name=command_name,
            file_path=str(file_path),
            description=description,
            category=category,
            tags=tags,
            content=content,
        )
