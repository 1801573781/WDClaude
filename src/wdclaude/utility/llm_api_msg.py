import json
from pathlib import Path

from wdclaude.paths import paths as _paths
from wdclaude.utility.skill_loader import get_skill_loader

# 越界允许目录列表持久化文件（B/.wdclaude/allowed_roots.json，跨会话生效）
_ALLOWED_ROOTS_FILE = _paths.data_root / "allowed_roots.json"


class ProjectRootManager:
    """project_root 沙箱管理器（模块级单例）。

    - /init 时设置 project_root 并加载对应目录树（每个 project_root 仅加载一次，
      刷新后的 .tree_cache.md 不重复加载）
    - /init 之外，代码增/删/改操作默认限制在 project_root 之下；
      越界时由 tool_executor 守卫层弹出 AskUserQuestion 询问用户，确认后才继续
    - project_root 为空（未执行 /init）时，越界操作同样需用户确认后才继续
    - 用户在询问中选择"以后该目录都允许"时，该目录加入永久允许列表
      （B/.wdclaude/allowed_roots.json，跨会话持久化），
      后续该目录下的越界操作直接放行、不再询问
    """


    def __init__(self):
        self.project_root: Path | None = None

        # 目录树注入状态：{project_root 字符串: 是否已注入}
        self._tree_injected: dict[str, bool] = {}

        # 记录注入时的 .tree_cache.md mtime，用于判断文件是否刷新
        self._tree_injected_mtime: dict[str, float] = {}

        # 用户显式允许的越界目录列表（绝对路径字符串，跨会话持久化）
        self._allowed_roots: list[str] = []
        self._load_allowed_roots()


    def set_project_root(self, project_root) -> None:
        """设置 project_root（/init 执行时调用），并加载其目录树。"""
        root_path = Path(project_root).expanduser().resolve()
        self.project_root = root_path


    def inject_tree_cache(self, api_messages) -> bool:
        """根据 project_root 加载对应 .wdclaude/init/<project_root>/.tree_cache.md
        并注入到 api_messages 尾部。

        加载策略（只加载一次）：
        - 每个 project_root 仅加载一次，注入后不再重复加载
        - 已加载且 .tree_cache.md 未刷新 → 不再加载
        - 已加载且 .tree_cache.md 有刷新 → 也不再加载

        Returns:
            bool: 本次是否实际执行了注入
        """
        if self.project_root is None:
            return False

        root_key = str(self.project_root)

        # 已注入过 → 不再加载（无论 .tree_cache.md 是否刷新）
        if self._tree_injected.get(root_key):
            return False

        tree_path = _paths.init_dir_for(self.project_root) / ".tree_cache.md"
        try:
            if not tree_path.is_file():
                return False
            tree_cache = tree_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"[warn] 加载 .tree_cache.md 失败: {e}")
            return False

        api_messages.append_micro_info("system", f"[项目目录树]\n{tree_cache}")
        self._tree_injected[root_key] = True
        try:
            self._tree_injected_mtime[root_key] = tree_path.stat().st_mtime
        except OSError:
            pass
        return True


    def _load_allowed_roots(self) -> None:
        """从 B/.wdclaude/allowed_roots.json 加载允许目录列表。

        文件不存在或损坏时视为空列表（允许列表清零，不影响其他功能）。
        """
        try:
            if _ALLOWED_ROOTS_FILE.is_file():
                data = json.loads(_ALLOWED_ROOTS_FILE.read_text(encoding="utf-8"))
                roots = data.get("allowed_roots", [])
                if isinstance(roots, list):
                    self._allowed_roots = [str(r) for r in roots if str(r).strip()]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as e:
            print(f"[warn] 加载 allowed_roots.json 失败: {e}")
            self._allowed_roots = []


    def _save_allowed_roots(self) -> None:
        """持久化允许目录列表到 B/.wdclaude/allowed_roots.json。"""
        try:
            _ALLOWED_ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _ALLOWED_ROOTS_FILE.write_text(
                json.dumps({"allowed_roots": self._allowed_roots},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[warn] 保存 allowed_roots.json 失败: {e}")


    def add_allowed_root(self, path_str: str) -> str:
        """将目标路径所在目录加入允许列表（目标本身是目录则直接加入）。

        Args:
            path_str: 目标路径（文件或目录）。已是存在的目录 → 取自身；
              否则（文件/尚不存在的路径）→ 取父目录。

        Returns:
            实际加入（或已存在）的允许目录字符串；路径无法解析时返回空串
        """
        try:
            p = Path(path_str)
            if p.is_dir():
                allowed_dir = p.resolve()
            else:
                allowed_dir = p.parent.resolve()
        except (OSError, RuntimeError, ValueError):
            return ""
        key = str(allowed_dir)
        if key and key not in self._allowed_roots:
            self._allowed_roots.append(key)
            self._save_allowed_roots()
        return key


    def is_path_in_allowed_roots(self, path_str: str) -> bool:
        """检查路径是否位于任一允许目录之下（含目录本身）。

        允许列表为空或路径无法解析时返回 False（保守判定）。
        """
        if not self._allowed_roots:
            return False
        try:
            p = Path(path_str).resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        for root_str in self._allowed_roots:
            try:
                p.relative_to(root_str)
                return True
            except ValueError:
                continue
        return False


    def check_path_allowed(self, path_str: str) -> tuple[bool, str]:
        """检查路径是否位于 project_root 之下（纯路径位置判定）。

        project_root 为空 → 不允许
        相对路径默认位于 project_root 之下；绝对路径必须位于 project_root 之下。

        注意：本方法只负责位置判定，不与用户交互。"越界时先询问用户、
        确认后才继续"的策略由 tool_executor 的
        _guard_code_write_path / _guard_bash_delete 守卫层实现。

        Returns:
            (allowed, reason): allowed=True 表示位于 project_root 之下（或相对路径）；
            否则 reason 为不允许的原因描述（供守卫层或调用方使用）
        """
        if self.project_root is None:
            return False, (
                "project_root 未设置（尚未执行 /init）。"
                "请先执行 /init [--proj_root <路径>] 设置 project_root。"
            )

        p = Path(path_str)
        if not p.is_absolute():
            # 相对路径默认位于 project_root 之下 → 放行
            return True, ""

        # 绝对路径：必须位于 project_root 之下
        try:
            p.resolve().relative_to(self.project_root)
            return True, ""
        except ValueError:
            return False, (
                f"路径 {path_str} 不在 project_root ({self.project_root}) 之下。"
                f"project_root 之外的操作需用户确认后才可继续。"
            )


# 模块级单例，全局唯一
project_root_mgr = ProjectRootManager()


# 通过API与LLM交互时，Message的构建
class LLMAPIMessage:

    def __init__(self, role: str = "mycode"):
        # Phase 2（设计文档 P4/11.2）：占位符与目录树锚点从 A 根改为 B 根（cwd）。
        # Agent 模式下 B 是用户项目；Self 模式下 B 即开发仓库。
        self.project_root = _paths.b_root
        self.role = role

        # 从同目录加载 sys_prompt.md 作为系统提示词
        self.api_messages = self._load_system_prompt(role)

        # 注入项目上下文（放在系统提示词之后、用户输入之前）
        project_context = self._load_project_context(role)
        if project_context:
            context_msg = {
                "role": "system",
                "content": f"[项目上下文]\n{project_context}"
            }
            self._append_info(context_msg)

        # 504 变更：启动时不再加载 .tree_cache.md，
        # 目录树改由 /init 执行后按 project_root 注入（见 ProjectRootManager）
        self._initial_msg_count = len(self.api_messages)


    def _resolve_placeholders(self, text: str) -> str:
        """将文本中的 ${project_root} 占位符替换为实际项目根目录路径。
        统一使用正斜杠格式，确保与 config.yaml 中的路径风格一致。
        Phase 2：锚点为 B 根（设计文档 11.2，原为 A 根）。
        """
        root_str = str(self.project_root.resolve()).replace("\\", "/")
        return text.replace("${project_root}", root_str)


    def _find_role_file(self, role: str, filename: str):
        """两级资源查找（v2.0 第 7 章）：B/.wdclaude/config/role/<role>/ → 包内 defaults/roles/<role>/。"""
        for role_dir in _paths.role_dirs(role):
            cand = role_dir / filename
            if cand.is_file():
                return cand
        return None

    def _load_system_prompt(self, role: str) -> list[dict]:
        """三级查找加载系统提示词（7.3），包装为 API 消息格式，并追加技能清单（L1 Metadata）。"""
        md_path = self._find_role_file(role, f"{role}_sys_prompt.md")

        base_prompt = ""
        if md_path is not None:
            base_prompt = md_path.read_text(encoding="utf-8")
            base_prompt = self._resolve_placeholders(base_prompt)

        # 追加 L1 技能清单（如果有）
        skill_loader = get_skill_loader()
        skills_section = skill_loader.format_skills_prompt()
        if skills_section:
            base_prompt = base_prompt + "\n\n" + skills_section

        return [{"role": "system", "content": base_prompt}]


    def _load_project_context(self, role: str) -> str:
        """两级查找加载项目上下文（v2.0 第 7 章）：B/.wdclaude/wdclaude.md → 角色默认上下文。"""
        try:
            # B 项目级项目上下文（启动目录的项目说明，可选）
            b_ctx = _paths.b_wdclaude_md
            if b_ctx.exists():
                content = b_ctx.read_text(encoding="utf-8")
                return self._resolve_placeholders(content)

            # 角色默认项目上下文（两级查找：B/.wdclaude/config/role → 包内 defaults）
            md_path = self._find_role_file(role, f"{role}.md")
            if md_path is not None:
                content = md_path.read_text(encoding="utf-8")
                return self._resolve_placeholders(content)
            return ""
        except (OSError, UnicodeDecodeError) as e:
            print(f"[warn] 加载项目上下文失败: {e}")
            return ""


    @staticmethod
    def _load_tree_cache() -> str:
        """[废弃] 启动时不再加载目录树（504 变更），保留方法仅为兼容旧调用方。"""
        return ""


    def get_msg(self):
        return self.api_messages

    def reset_context(self):
        """重置上下文：保留初始化时的系统级消息（system prompt + 项目上下文 + 目录树），
        清空其后的所有对话历史。
        """
        self.api_messages = self.api_messages[:self._initial_msg_count]


    def init_api_msg(self, user_input):
        """
        初始化发送给 LLM 的 messages 列列：复制系统提示词，追加用户消息。
        """
        msg = {
            "role": "user",
            "content": user_input
        }

        self._append_info(msg)

    def build_command_messages(
        self,
        command_name: str,
        command_content: str,
        user_argument: str,
        project_context: str,
        directory_tree: str,
    ) -> list[dict[str, str]]:
        """组装斜杠命令的 api_messages

        采用方案 A（推荐）：命令内容作为会话重置。
        system 消息 = 系统提示词 + 命令指令 + 项目上下文 + 目录树，
        user 消息 = 用户参数（或默认提示语）。

        Args:
            command_name: 斜杠命令名，如 "/opsx:propose"
            command_content: 命令 .md 文件的正文内容
            user_argument: 用户在命令后输入的参数
            project_context: 项目上下文文本
            directory_tree: 目录树文本

        Returns:
            组装好的 api_messages 列表
        """
        messages = []

        # system 消息：系统提示词 + 命令指令 + 项目上下文 + 目录树
        system_prompt = self._load_system_prompt(role=self.role)[0]["content"]
        command_prompt = f"# 斜杠命令: {command_name}\n\n{command_content}"
        parts = [system_prompt, command_prompt]
        if project_context:
            parts.append(project_context)
        if directory_tree:
            parts.append(directory_tree)
        messages.append({
            "role": "system",
            "content": "\n\n".join(parts),
        })

        # user 消息：用户参数（或默认提示语）
        if user_argument.strip():
            messages.append({
                "role": "user",
                "content": user_argument,
            })
        else:
            messages.append({
                "role": "user",
                "content": "请按照上述命令指令开始执行。",
            })

        return messages

    def reset_with_command(
        self,
        command_name: str,
        command_content: str,
        user_argument: str,
    ) -> None:
        """用斜杠命令上下文重置 api_messages（方案 A：会话重置）

        Args:
            command_name: 斜杠命令名
            command_content: 命令 .md 正文内容
            user_argument: 用户参数
        """
        # 504 变更：斜杠命令上下文不再注入目录树（目录树仅由 /init 后注入）
        project_context = self._load_project_context(self.role)

        new_messages = self.build_command_messages(
            command_name=command_name,
            command_content=command_content,
            user_argument=user_argument,
            project_context=project_context,
            directory_tree="",
        )

        self.api_messages = new_messages

    # 尾部添加微信息
    def append_micro_info(self, role, micro_info):
        """
        代码看起来很简单，就是 api_messages.append(micro_info)，但是这里蕴含着一种思想：
        如果添加的是微信息，那么就不涉及对 api_messages 中的 memory 的压缩等操作，
        直接将微信息 append 到 api_messages 尾部即可
        """
        msg = {
            "role": role,
            "content": micro_info
        }

        self._append_info(msg)

    # 尾部添加微 LLM 的 response
    def append_llm_response(self, llm_response):
        """
        1、现在，先简单地：api_messages.append(llm_response)
        2、后续，要考虑转记忆、压缩等等
        """
        msg = {
            "role": "assistant",
            "content": llm_response
        }

        self._append_info(msg)

    # 尾部添加 tool 执行结果
    def append_tool_exec_result(self, result_msg):
        # 因为result_msg直接就是 dict，所以直接append即可
        self._append_info(result_msg)

    # 这是一个内部的简化实现
    def _append_info(self, msg):
        self.api_messages.append(msg)

    def insert_before_last_user(self, content):
        """在最后一条 user 消息之前插入一条 user 消息（视觉 B 注入用，multimodal_spec 4.7）。

        消息顺序契约：会话级说明 -> B（视觉提取） -> A（用户原始输入）。
        A 已通过 init_api_msg append 为末尾 user 消息；调用方按
        "先插说明、再插 B" 的顺序调用两次本方法，即可得到
        [说明, B, A] 的正确排布（每次插入都落在当前最后一条 user 之前）。
        找不到 user 消息时降级为普通 append。
        """
        insert_at = None
        for i in range(len(self.api_messages) - 1, -1, -1):
            if self.api_messages[i].get("role") == "user":
                insert_at = i
                break
        msg = {"role": "user", "content": content}
        if insert_at is None:
            self._append_info(msg)
        else:
            self.api_messages.insert(insert_at, msg)
