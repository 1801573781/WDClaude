from enum import Enum
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Callable, Dict
from wdclaude.query import chat_llm
from wdclaude.utility import llm_api_msg
from wdclaude.llm_tool import tool_executor
from wdclaude.utility.config_loader import global_cfg
from wdclaude.utility.normal_utility import strip_thinking
from wdclaude.query.session_log import SessionLog
from wdclaude.query.todo_manager import TodoManager
from wdclaude.memory_ex.factory import create_memory
import logging
from wdclaude.utility.token_statistics import record_token_usage
import re
import hashlib
from wdclaude.memory_ex.memory_interface import NoopMemory
from wdclaude.utility.token_statistics import get_token_summary
from wdclaude.utility.shutdown import shutdown_mgr
import copy

logger = logging.getLogger(__name__)


class ChatOrNot(Enum):
    QuitByNoneTool = 1
    QuitByDone = 2
    Continue = 3


class QueryLoop:
    """
    与 LLM 的多轮交互引擎。
    不负责任何终端显示，只通过回调接口通知外部。
    """


    def __init__(self, show_thinking: bool = False, role: str = "mycode"):
        self.show_thinking = show_thinking
        self.role = role

        self.api_messages = llm_api_msg.LLMAPIMessage(role=self.role)
        self.session = SessionLog()
        self.max_turns = 0
        self.is_multi_turns = None
        self._query_counter = 0

        # 通过工厂函数创建记忆实例（根据 config.yaml memory.backend 选择后端）
        self._init_memory()
        self._memory_used = False

        self._print_info = None
        self._print_llm_rsp = None
        self._print_llm_reasoning = None
        self._print_tool_call = None
        self._print_tool_result = None

        # 精确 token 统计（从 API usage 获取，非粗略估算）
        self.prompt_cache_hit = 0    # 输入（命中缓存）
        self.prompt_cache_miss = 0   # 输入（未命中缓存）
        self.completion_tokens = 0   # 输出

        # 追问无工具兜底计数器
        self._no_tool_retry = 0

        # 斜杠命令上下文（None 表示普通对话模式）
        self._command_context = None

        # 死循环熔断：记录上一轮的工具签名（连续两轮工具列表完全一致时强制终止）
        self._last_tool_sig = None

        # TodoWrite 有状态进度管理
        self._todo_manager = TodoManager()
        self._on_todo_update = None  # 回调：todo 列表更新时通知 CLI 渲染
        self._last_todo_sig = None  # 死循环熔断：上一轮 todo 状态签名

        # 延迟记忆召回：当用户输入引用了文件时，推迟到文件读取后再召回
        # （仅 L1 不可用/回退模式使用；正常模式下由 L0/L1 分流取代，multimodal_spec 3.1）
        self._pending_memory_recall = False
        self._current_user_input = ""

        # ===== L0/L1 输入分流（multimodal_spec 3.1，惰性装配，零侵入）=====
        # 媒体注册表（media.handlers 配置驱动 D10）：None=未装配，装配失败置 _vision_registry_failed
        self._vision_registry = None
        self._vision_registry_failed = False
        # 会话级视觉能力说明是否已注入（4.6，仅首次多模态注入一次）
        self._vision_notice_injected = False
        # 输入分流 fail-fast 标志：现仅 view_image 工具级致命错误置位；
        # L1 意图分类与视觉提取失败均已改为规则降级（不终止用户输入）
        self._input_route_fatal = False

        # 延迟记忆召回日志：file_view 后触发的召回结果，推迟到下一 Turn 记录
        self._pending_retrieval_result = None

        # file_view 重复查看防护：同一文件累计查看第 3 次注入最终警告，第 4 次起硬阻断
        self._file_view_counts = {}  # {文件路径: 查看次数}

        # ===== 优雅退出（阶段二） =====
        # Query 收尾幂等标志：run() 开始时置 False，_finalize_query() 执行后置 True
        self._query_finalized = True
        # Turn 记忆保存标志：正常路径保存后置 True（中断补存判据）
        self._turn_memory_saved = True
        # 当前 Turn 序号缓存：供 _finalize_query() 记录延迟召回日志
        self._current_loop_turn = 0


    def _init_memory(self):
        """通过工厂函数创建记忆实例，容错降级为 NoopMemory。"""
        try:
            self._memory = create_memory(global_cfg)
            logger.info(f"记忆模块初始化成功: {type(self._memory).__name__}")
            # 注入 LLM 调用函数，供记忆系统的提取器/整理器/进化器使用
            if hasattr(self._memory, 'set_llm_chat_fn'):
                #  from wdclaude.query import chat_llm
                self._memory.set_llm_chat_fn(chat_llm.simple_chat)
        except Exception as e:
            logger.warning(f"记忆模块初始化失败，降级为 NoopMemory: {e}")
            #  from wdclaude.memory_ex.memory_interface import NoopMemory
            self._memory = NoopMemory()


    def clear_memory(self) -> dict:
        """清除所有持久化记忆（Layer 0/1/2 + 元数据）。

        仅清除记忆系统的持久化数据，不影响当前会话上下文（api_messages）。
        下次新 Query 开始时，记忆检索将返回空结果，LLM 不再获得旧记忆。

        Returns:
            清除统计字典，包含各层清除的条目数等信息；若记忆模块未启用，返回空 dict
        """
        if self._memory is None:
            return {}
        result = self._memory.clear_all()
        if isinstance(result, dict):
            return result
        # 兼容旧后端仍返回 int 的情况
        return {"total": result} if isinstance(result, int) else {}

    def reset_context(self):
        """重置上下文：保留 system prompt 及初始化消息，清空对话历史。
        保留此方法用于内部调用，CLI 层不再直接映射到 /r ctx 命令。
        """
        if self.api_messages:
            self.api_messages.reset_context()
            logger.info("上下文已重置（保留系统提示词，清空对话历史）")


    def new_session(self):
        """开启一个新的 Session：清空 api_messages（仅保留 system_prompt）、
        创建新的 SessionLog 实例。

        注意：记忆是跨 Session 的持久化数据，新开 Session 不清除记忆。
        如需清除记忆，请使用 /r mem 命令。
        """
        # 1. 重置 api_messages（仅保留 system_prompt）
        self.api_messages.reset_context()

        # 2. 创建新的 SessionLog 实例
        self.session = SessionLog()

        # 3. 重置 Query 计数器
        self._query_counter = 0

        # 4. 重置 token 统计
        self.prompt_cache_hit = 0
        self.prompt_cache_miss = 0
        self.completion_tokens = 0

        # 5. 重置视觉运行时单例（view_image 会话级熔断计数归零，multimodal_spec 7.4）
        try:
            from wdclaude.vision import reset_vision_runtime
            reset_vision_runtime()
        except Exception as e:
            logger.warning(f"重置视觉运行时失败: {e}")

        # 6. 重置 web_search / web_fetch 会话级调用计数（spec 9.3 L2）
        try:
            from wdclaude.tools.web_search import reset_session_counters
            reset_session_counters()
        except Exception as e:
            logger.warning(f"重置 web 会话计数失败: {e}")

        logger.info("新 Session 已开启（上下文已重置，记忆保留）")


    def record_cli_command(self, command: str):
        """记录 CLI 命令到 session log 和 api_messages。

        Args:
            command: 用户输入的 CLI 命令（如 /test --st-e）
        """
        self.api_messages.append_micro_info("system", f"[CLI_COMMAND] {command}")
        self.session.log_cli_command(command)

    def append_cli_result(self, result_summary: str):
        """记录 CLI 命令执行结果摘要到 session log 和 api_messages。

        Args:
            result_summary: 命令执行结果的精炼摘要或完整输出
        """
        self.api_messages.append_micro_info("system", f"[CLI_RESULT] {result_summary}")
        self.session.log_cli_result(result_summary)

    def get_tokens(self):
        """返回详细的 token 统计字典（从持久化统计文件读取）。
        keys: prompt_cache_hit, prompt_cache_miss, completion_tokens, total
        """
        #  from wdclaude.utility.token_statistics import get_token_summary
        return get_token_summary()


    def run(
            self,
            user_input: str,
            on_context_mgr: Callable[[str], AbstractContextManager],
            print_info: Callable[[str], None],
            print_llm_rsp: Callable[[str], None],
            print_tool_call: Callable[[str, Dict], None],
            print_tool_result: Callable[[str, str, dict | None], None],
            print_llm_reasoning: Callable[[str, int], None] = None,
            command_context: dict | None = None,
            on_todo_update: Callable[[dict], None] = None,
    ):

        # 赋值
        self._print_info = print_info
        self._print_llm_rsp = print_llm_rsp
        self._print_tool_call = print_tool_call
        self._print_tool_result = print_tool_result
        self._print_llm_reasoning = print_llm_reasoning
        self._on_todo_update = on_todo_update

        """
        每一次 query_loop.run 的调用，都是与 LLM 的一次新 Query。
        api_messages 跨 Query 累积，不在此处重置。
        只有 /new session 才会清空上下文。
        """
        turn = 0
        quit_chat = ChatOrNot.Continue

        self._query_counter += 1
        self.session.start_query(self._query_counter, user_input)
        self._current_query_text = user_input  # 供追问 token 记录使用
        # 记录最近一条用户消息，供 web_search 的 ori_question 自动附加（spec 9.4）
        try:
            from wdclaude.tools import web_search as _ws
            _ws.set_last_user_input(user_input)
        except Exception:
            pass
        self._query_finalized = False  # 本 Query 收尾待执行（_finalize_query 幂等完成）
        self._turn_memory_saved = False  # Turn 记忆待保存（中断补存判据）
        self.max_turns = global_cfg.cli.max_turns
        self._no_tool_retry = 0  # 每次新 Turn 重置追问计数器
        self.is_multi_turns = None  # 每次 Turn 开始时未确定
        self._todo_manager.reset()  # 每次新 Turn 重置 todo 列表
        self._pending_memory_recall = False  # 重置延迟召回标志
        self._current_user_input = ""  # 清空缓存的用户输入
        self._input_route_fatal = False  # 重置输入分流 fail-fast 标志（multimodal_spec D8）
        self._pending_retrieval_result = None  # 清空延迟召回日志缓存
        self._file_view_counts = {}  # 重置 file_view 重复查看计数器
        self._last_tool_sig = None  # 重置工具签名熔断器（跨 Query 不比对，防止误熔断）

        # 如果有斜杠命令上下文，用命令内容重置 api_messages
        self._command_context = command_context
        if command_context:
            self.api_messages.reset_with_command(
                command_name=command_context["command_name"],
                command_content=command_context["command_content"],
                user_argument=command_context["user_argument"],
            )

        # 新任务开始，执行记忆维护（遗忘过期记忆，不删除持久化数据）
        try:
            self._memory.maintain()
        except Exception as e:
            logger.warning(f"记忆维护失败: {e}")

        # 开始跟LLM多次循环交互，目的是为了完成用户任务（user_input）
        while turn < self.max_turns:
            # ===== 安全点（优雅退出阶段二）：Turn 边界轮询中断标志 =====
            if shutdown_mgr.is_interrupted():
                self._print_info("[优雅退出] 收到中断请求，任务在 Turn 边界安全终止")
                self.session.log_dict_info({
                    "role": "system",
                    "content": f"[优雅退出] 用户中断（{shutdown_mgr.interrupt_reason}），"
                               f"任务于 Turn {turn} 边界安全终止"
                })
                quit_chat = ChatOrNot.QuitByNoneTool
                break

            turn += 1
            self._current_loop_turn = turn  # 供 _finalize_query 记录延迟召回日志

            """1、发送请求给LLM"""
            with on_context_mgr(f"Thinking-{turn}"):  # 显示“thinking”在闪烁，表明系统未死，只是在等LLM的回复
                # 发送请求给LLM，前期准备
                thinking_begin = self._on_llm_req(turn, user_input)

                # ===== 输入分流 fail-fast 兜底检查（L1/视觉失败已降级，正常不触发）=====
                if self._input_route_fatal:
                    quit_chat = ChatOrNot.QuitByNoneTool
                    break

                # 发送请求给LLM
                # ===== 优雅退出：流式等待期间被中断时断流退出 =====
                # （break 经 with 上下文正常退出，Thinking 终端状态还原）
                try:
                    ai_response, is_truncated, reasoning_content, usage = chat_llm.chat_with_retry(self.api_messages.get_msg())  # noqa E501
                except KeyboardInterrupt:
                    shutdown_mgr.mark_interrupt()
                    self._print_info("[优雅退出] 收到中断请求，停止等待 LLM 响应")
                    self.session.log_dict_info({
                        "role": "system",
                        "content": "[优雅退出] 等待 LLM 响应时被用户中断，已断流"
                    })
                    quit_chat = ChatOrNot.QuitByNoneTool
                    break

                # 累积精确 token 统计（内存计数器，用于实时显示）
                if usage:
                    cached = usage.get("cached_tokens", 0)
                    prompt_total = usage.get("prompt_tokens", 0)
                    self.prompt_cache_hit += cached
                    self.prompt_cache_miss += (prompt_total - cached)
                    self.completion_tokens += usage.get("completion_tokens", 0)

                    # 持久化 token 统计到 CSV 文件
                    record_token_usage(
                        model_name=chat_llm.model_name,
                        prompt_tokens=prompt_total,
                        cached_tokens=cached,
                        completion_tokens=usage.get("completion_tokens", 0),
                        query=user_input,
                        turn=f"turn{turn}",
                    )

                # ===== Prompt 超长熔断（415 现场）=====
                # 上下文膨胀超过模型上限时 API 直接报错，消息列表只会更长，
                # 任何重试/追问都必然失败（415 现场：追问 3 次全部返回同样的
                # 超长错误，白白多烧 3 次调用与等待）。检测到超长错误立即终止本 Turn。
                if ai_response.startswith("[API_ERROR:") and any(
                    kw in ai_response.lower()
                    for kw in ("max length", "context length", "context_length",
                               "too long", "too large", "prompt is too")
                ):
                    self._print_info(
                        "[上下文超限] 对话上下文已超过模型最大长度限制，重试必然失败，本任务强制终止。\n"
                        "建议：1) 输入 /new session 开启新会话后继续任务；"
                        "2) 检查是否存在重复读取文件、工具结果过大等问题。"
                    )
                    self.session.log_dict_info({
                        "role": "system",
                        "content": "[上下文超限熔断] API 返回 Prompt 超长错误，强制终止本 Turn"
                    })
                    quit_chat = ChatOrNot.QuitByNoneTool
                    break

            """2. 解构 LLM response"""
            tools, remaining_text = self._on_llm_rsp(turn, thinking_begin,
                                                     ai_response, reasoning_content)

            """3. 开始处理工具"""
            quit_chat, tool_exec_info = self._handle_tools(tools, remaining_text)

            # 只记录首个完整 Turn 的记忆（避免同一 Query 多 Turn 重复写入 raw 记忆）
            # 后续 Turn 主要是工具执行跟进，核心信息已在首 Turn 中体现
            if self._memory_used and user_input and turn == 1:
                self._save_turn_memory(turn, user_input, reasoning_content,
                                       remaining_text, tool_exec_info)

            if not quit_chat == ChatOrNot.Continue:  # 如果不是继续chat，那就break循环
                break

        # 退出了循环
        if turn >= self.max_turns and quit_chat == ChatOrNot.Continue:  # 这种情况表明，实际上LLM并没有找到正确答案，但是，强制退出了
            self._print_info(f"达到最大轮次限制 ({self.max_turns})，强制结束")
        # 否则的话，就是正常退出，这里不用打印任何信息

        # ===== 优雅退出（阶段二）：Query 收尾统一入口（幂等） =====
        # 正常退出 / Turn 边界中断 / 流式断流 / CLI 层 finally 兜底（finalize_query）
        # 全部汇聚到 _finalize_query，确保日志落盘与中断记忆补存只执行一次
        self.finalize_query()

        # Query 结束时：执行记忆生命周期维护（提取 → 自动整理/进化 + 提示）
        # maintain() 已在新任务开始时执行，此处不再重复
        # 【优雅退出】中断场景跳过——含 LLM 调用的耗时操作，退出时刻不再等待
        if self._memory_used and not shutdown_mgr.is_interrupted():
            try:
                # 1. 维护：检查水位、衰减评分（轻量，不做整理）
                self._memory.maintain()

                # 2. 自动整理（如果配置开启）
                try:
                    chat_llm.set_context(query="auto_compaction", turn="CLI_COMMAND")
                    self._memory.auto_compact()
                    chat_llm.set_context()
                except Exception as e:
                    chat_llm.set_context()
                    logger.warning(f"自动整理失败: {e}")

                # 3. 自动进化（如果配置开启）
                try:
                    chat_llm.set_context(query="auto_evolution", turn="CLI_COMMAND")
                    self._memory.auto_evolve()
                    chat_llm.set_context()
                except Exception as e:
                    chat_llm.set_context()
                    logger.warning(f"自动进化失败: {e}")

                # 4. 提示机制
                # 4.1 整理与进化提示（如果自动关闭且需要执行）
                if self._memory.check_compaction_needed():
                    self._print_info("记忆需要整理，执行 /mem compaction 命令整理记忆")
                if self._memory.check_evolution_needed():
                    self._print_info("记忆需要进化，执行 /mem evolution 命令进化记忆")

                # 4.2 记忆提取提示（提取为手动命令 /mem extract）
                # 空库场景（MEMORY.md 为空）下 4.1 的两个提示永不触发，
                # MD 会话日志会无限累积而无人提醒——由此补充提取引导，
                # 阈值 10 对齐旧版 raw_prompt_threshold 语义
                try:
                    mem_stats = self._memory.stats()
                    md_pending = mem_stats.get("md_pending", 0)
                    if md_pending >= 10:
                        self._print_info(
                            f"已积累 {md_pending} 个会话日志待提取，执行 /mem extract 命令提取记忆"
                        )
                except Exception as e:
                    logger.warning(f"记忆提取提示检查失败: {e}")

                logger.info("记忆生命周期维护完成")
            except Exception as e:
                logger.error(f"记忆生命周期处理失败: {e}")


    def finalize_query(self):
        """Query 收尾幂等入口（公开方法，供 CLI 层 finally 兜底调用）。

        无论 run() 从哪条路径离开（正常 done / 最大轮次 / Turn 边界中断 /
        流式断流 / 异常冒泡由外层 finally 调用），日志落盘与中断记忆补存
        在此只执行一次。
        """
        self._finalize_query(self._current_loop_turn, self._current_query_text,
                             interrupted=shutdown_mgr.is_interrupted())

    def _finalize_query(self, turn: int, user_input: str, interrupted: bool = False):
        """Query 收尾（内部幂等实现）：延迟召回落盘 + Turn flush + Query 关闭。

        原 run() 尾部的收尾逻辑迁移至此并幂等化：
        1. _query_finalized 标志保证多路径调用（正常路径 + CLI finally 兜底）
           只执行一次；
        2. 中断场景额外执行 Turn 记忆补存（标记 interrupted），确保被中断
           的任务在 raw 记忆层留下痕迹（完整内容已随 SessionLog 落盘，
           后续 /mem extract 仍可从 MD 日志提取）。
        """
        if self._query_finalized:
            return
        self._query_finalized = True

        try:
            # 兜底：延迟召回日志还未被消费时，先 flush 当前 Turn，
            # 再在新 Turn 中记录召回日志（避免记忆召回出现在触发的 Turn 中）
            if self._pending_retrieval_result is not None:
                self.session.flush_turn()
                self.session.log_turn(turn + 1)
                self.session.log_memory_retrieval(self._pending_retrieval_result)
                self._pending_retrieval_result = None

            # 确保最后一个 Turn 的内容被持久化，并关闭当前 Query
            self.session.end_query()
        except Exception as e:
            logger.error(f"Query 收尾日志落盘失败: {e}")

        # 中断场景：Turn 记忆补存（尽力完成级别）
        # 判据：中断 + 记忆启用 + 本 Query 尚未保存过 Turn 记忆
        # （_save_turn_memory 仅在 turn==1 末尾保存，中断发生在保存前才需补存）
        if interrupted and self._memory_used and not self._turn_memory_saved and user_input:
            try:
                reason = shutdown_mgr.interrupt_reason or "user"
                content = (
                    f"[用户输入] {user_input}\n\n"
                    f"[中断说明] 用户在 Turn {turn} 执行期间中断任务（{reason}），"
                    f"任务未完成。完整过程见会话日志。"
                )
                self._memory.add("", content, metadata={
                    "turn": turn,
                    "has_tools": True,
                    "has_reasoning": False,
                    "query_id": self._query_counter,
                    "user_input": user_input,
                    "session_id": getattr(self.session, 'session_id', ''),
                    "interrupted": True,
                })
                logger.info(f"中断 Turn 记忆已补存 (turn={turn})")
            except Exception as e:
                logger.warning(f"中断 Turn 记忆补存失败: {e}")


    """
    1. 本来想着，这里跟CLI那个模块，完全解耦，但是发现做不到
    2. 只能做到部分解耦：
       2.1 什么时候打印，打印什么内容，本该是CLI模块的内容，还是放在这里实现了
       2.2 打印的方法与技术，对这里透明，由CLI模块“注册”进来
    3. 纠结了好几天，最终决定与自己和解，不再追求完全解耦了
    """

    # ==================================================================
    # L0/L1 输入分流（multimodal_spec 3.1 / 3.2，D4/D6/D8/D9）
    # ==================================================================
    def _get_media_extensions(self) -> list:
        """惰性装配媒体注册表，返回已启用媒体的扩展名清单（D10 配置驱动）。

        装配失败（media.handlers 缺失等）降级为空清单：L0 退化为纯文本
        引用检测，媒体文件按普通候选走 L1，不影响主流程。
        """
        if self._vision_registry is not None or self._vision_registry_failed:
            return self._vision_registry.media_extensions if self._vision_registry else []
        try:
            from wdclaude.vision.media_types import MediaRegistry
            self._vision_registry = MediaRegistry.from_config(global_cfg.media)
            logger.info(f"媒体注册表装配完成: {self._vision_registry.describe()}")
            return self._vision_registry.media_extensions
        except Exception as e:
            logger.warning(f"媒体注册表装配失败，媒体引用按文本候选处理: {e}")
            self._vision_registry_failed = True
            return []

    def _vision_config(self, key: str, default):
        """安全读取 vision 配置项（段缺失时返回默认值）。"""
        vision_cfg = getattr(global_cfg, "vision", None)
        return getattr(vision_cfg, key, default) if vision_cfg is not None else default

    def _route_input_references(self, query_text: str):
        """L0 检测 -> L1 意图分类 -> 按角色分岔（multimodal_spec 3.1 总流程）。

        静默执行（用户不感知分类过程），返回：
        - recall_query: 记忆召回查询文本（A + L1 summary + 引用角色清单 + B 摘要）
        - preread_paths: requirement_doc 角色路径清单（召回后由调用方预读注入）
        - b_message: 视觉 B 消息文本（无 media_input 时为 None）

        分岔规则（3.1）：
        - L0 零候选 -> 纯文字路径（零侵入，返回原文）
        - 有候选 -> L1 分类（simple_chat 单轮旁路，重试 2 次）
          a) 含 media_input 且 vision.enabled -> 视觉提取 B -> 召回查询增强
          b) requirement_doc -> 收集预读清单（召回后预读）
          c) operation_target -> 不预读不延迟（直接召回，解决"莽撞"问题）
        - vision.enabled=false / intent_classify_enabled=false -> 回退旧行为
          （候选一律按需求文档：延迟召回 _pending_memory_recall=True）
        - L1 失败 -> 规则降级分类（媒体扩展名->media_input，其余->operation_target），
          任务照常进行（分类是便利性优化，降级而非终止用户输入）
        - 视觉提取失败 -> 降级为 view_image 引导提示（主循环可自行阅读图像）

        本方法不再置 _input_route_fatal（原 D8/D9 fail-fast 已废止）。
        """
        from wdclaude.vision.input_resolver import resolve_input

        media_exts = self._get_media_extensions()
        resolved = resolve_input(query_text, media_exts)

        # L0 零候选：纯文字路径，零侵入（3.4）——不进 L1、不碰视觉
        if not resolved.has_references:
            return query_text, [], None

        candidates = [ref.path for ref in resolved.references]

        # vision 总开关 / L1 开关关闭 -> 回退旧行为（候选一律按需求文档延迟召回）
        if not self._vision_config("enabled", True) or not self._vision_config("intent_classify_enabled", True):
            self._pending_memory_recall = True
            self._current_user_input = query_text
            self.session.log_dict_info({
                "role": "system",
                "content": f"[输入分流] vision/intent_classify 已关闭，回退旧行为：候选引用 {candidates} 按需求文档延迟召回"
            })
            return query_text, [], None

        # L1 意图分类（simple_chat 单轮旁路，静默，3.2）
        try:
            from wdclaude.vision.intent_classifier import classify as intent_classify
            from wdclaude.vision.intent_classifier import IntentClassifyFatalError

            def _llm_call(user_message: str) -> str:
                from wdclaude.vision.intent_classifier import load_prompt
                # simple_chat 为单 user 消息接口（无 system 参数）：分类准则
                # 拼接在 user 消息之前（L1 低温单轮调用，spec 3.2）
                full_prompt = load_prompt() + "\n\n" + user_message
                return chat_llm.simple_chat(full_prompt, temperature=0.1, max_tokens=2048)

            intent = intent_classify(resolved.clean_text or query_text, candidates, _llm_call)
        except IntentClassifyFatalError as e:
            # 规则降级替代终止（原 D8 fail-fast 已废止）：L1 是便利性优化
            # （决定预读/视觉提取/召回增强），失败时按扩展名规则分类即可，
            # 绝不能因分类失败丢弃用户的整轮输入
            intent = self._fallback_intent_result(candidates)
            self._print_info(
                f"[输入分流] 意图分类失败，已降级为按文件类型分类"
                f"（图片视为图像输入，其余视为操作对象），任务继续。原因: {str(e)[:120]}"
            )
            self.session.log_dict_info({
                "role": "system",
                "content": f"[输入分流-降级] L1 失败: {e}；规则分类: {intent.role_list_text()}"
            })
        except Exception as e:
            # simple_chat 本身异常（网络等）：同样规则降级（任务继续优于终止）
            intent = self._fallback_intent_result(candidates)
            self._print_info(
                f"[输入分流] 意图分类调用异常，已降级为按文件类型分类，任务继续。原因: {str(e)[:120]}"
            )
            self.session.log_dict_info({
                "role": "system",
                "content": f"[输入分流-降级] L1 调用异常: {e}；规则分类: {intent.role_list_text()}"
            })

        # 分类成功或规则降级：记录角色裁定（日志审计）
        self.session.log_dict_info({
            "role": "system",
            "content": f"[输入分流] L1 角色裁定: {intent.role_list_text()} | summary: {intent.summary}"
        })

        # 收集 requirement_doc 预读清单（召回后注入；operation_target 不预读）
        preread_paths = [r.path for r in intent.requirement_refs]

        # media_input 触发视觉提取链路（4.1）
        b_message = None
        b_summary = ""
        if intent.media_refs and self._vision_config("enabled", True):
            # 视觉提取失败时内部已降级为 view_image 引导（不终止本轮）
            b_message, b_summary = self._run_vision_extraction(
                [r.path for r in intent.media_refs],
                pure_image=not (resolved.clean_text or "").strip(),
            )

        # 召回查询 = A + summary + 引用角色清单（+ B 摘要，spec 3.1）
        recall_parts = [query_text]
        if intent.summary:
            recall_parts.append(f"[意图] {intent.summary}")
        role_list = intent.role_list_text()
        if role_list:
            recall_parts.append(f"[引用] {role_list}")
        if b_summary:
            recall_parts.append(f"[视觉] {b_summary}")
        recall_query = "\n".join(recall_parts)
        return recall_query, preread_paths, b_message

    def _fallback_intent_result(self, candidates: list):
        """L1 意图分类失败时的规则降级（扩展名启发式，零 LLM 依赖）。

        降级策略（安全性优先）：
        - 媒体扩展名（注册表启用项）-> media_input：视觉提取为只读操作，
          失败时另有 view_image 引导兜底
        - 其余 -> operation_target：不预读不延迟，主循环工具按需读取，
          零误读风险（与 L1 prompt "拿不准时倾向 operation_target" 一致）

        设计依据：L1 是便利性优化（决定预读/视觉提取/召回增强），失败时
        降级规则即可；终止用户整轮输入是体验事故，绝不可取。
        """
        from wdclaude.vision.intent_classifier import IntentResult, RefRole

        media_exts = {
            str(e).lstrip(".").lower()
            for e in self._get_media_extensions()
            if str(e).strip()
        }
        refs = []
        for p in candidates:
            ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
            role = "media_input" if ext in media_exts else "operation_target"
            refs.append(RefRole(path=p, role=role))
        return IntentResult(
            references=refs,
            is_multimodal=any(r.role == "media_input" for r in refs),
            summary="",
        )

    def _run_vision_extraction(self, media_paths: list, pure_image: bool):
        """视觉提取链路（multimodal_spec 4.1：预处理 -> 提取 -> B 渲染）。

        返回 (b_message, b_summary)；视觉运行时在此惰性装配（main_mlm 路由）。
        装配/提取失败降级为 view_image 引导提示（注入 micro_info，主循环可
        自行阅读图像），不再终止本轮。
        """
        try:
            from wdclaude.vision import get_vision_runtime, run_vision_pipeline
            from wdclaude.vision.vision_client import VisionFatalError

            # 共享进程级单例：与 view_image 工具统一熔断计数与 main_mlm 客户端
            runtime = get_vision_runtime()

            result, b_message = run_vision_pipeline(
                media_paths, runtime.registry, runtime.extractor,
                pure_image=pure_image,
            )
            self._print_info(f"[视觉提取] 已完成 {len(media_paths)} 张图像的结构化提取（main_mlm 路由）")
            return b_message, result.brief_summary()
        except VisionFatalError as e:
            # 降级而非终止：注入 view_image 引导，主循环可自行阅读图像
            # （view_image 自带错误处理与会话级熔断，失败原因可见可换图重试）
            self._print_info(f"[视觉提取] 自动提取失败，已降级为 view_image 引导（{e}）")
            self.session.log_dict_info({"role": "system", "content": f"[视觉-降级] {e}"})
            self.api_messages.append_micro_info(
                "system",
                f"[视觉提示] 图像自动提取失败（{str(e)[:200]}）。"
                f"如任务需要图像内容，请使用 view_image 工具逐张阅读: {', '.join(media_paths)}"
            )
            return None, ""
        except Exception as e:
            self._print_info(f"[视觉提取] 自动提取异常，已降级为 view_image 引导（{e}）")
            self.session.log_dict_info({"role": "system", "content": f"[视觉-降级] 异常: {e}"})
            self.api_messages.append_micro_info(
                "system",
                f"[视觉提示] 图像自动提取异常（{str(e)[:200]}）。"
                f"如任务需要图像内容，请使用 view_image 工具逐张阅读: {', '.join(media_paths)}"
            )
            return None, ""

    def _inject_preread(self, path: str):
        """需求文档预读注入（召回之后，spec 3.1 b 路径）。

        带行数上限（防超大文件撑爆上下文），注入文本明示截断情况，
        完整内容仍可由主模型按需 file_view。
        """
        try:
            from wdclaude.utility.file_tool import read_file
            max_lines = 500
            content = read_file(path, fallback="")
            if not content:
                self.api_messages.append_micro_info(
                    "system",
                    f"[预读失败] {path} 读取为空（不存在或编码异常），请用 file_view 确认",
                )
                return
            lines = content.splitlines()
            truncated = len(lines) > max_lines
            if truncated:
                content = "\n".join(lines[:max_lines])
                content += f"\n...（共 {len(lines)} 行，已截断至前 {max_lines} 行，完整内容可用 file_view 查看）"
            self.api_messages.append_micro_info(
                "system", f"[需求文档预读] {path}\n{content}"
            )
            self._print_info(f"[预读] 需求文档已注入: {path}（{len(lines)} 行）")
            self.session.log_dict_info({
                "role": "system",
                "content": f"[预读] {path} 已注入（{len(lines)} 行{'，截断' if truncated else ''}）"
            })

            # ===== F4：文档内嵌媒体引用检测（补 L0/L1 分流的嵌套引用盲区）=====
            media_paths = self._detect_embedded_media_refs(content)
            if media_paths:
                hint = (
                    f"[视觉提示] 需求文档 {path} 中引用了 {len(media_paths)} 个媒体文件：\n"
                    + "\n".join(f"- {m}" for m in media_paths)
                    + "\n阅读这些图片必须使用 view_image 工具（file_view 读图片只会得到二进制乱码）。"
                    "任务涉及图片内容时，请依次调用 view_image 阅读后再继续。"
                )
                self.api_messages.append_micro_info("system", hint)
                self._print_info(f"[视觉提示] 需求文档内嵌 {len(media_paths)} 个媒体文件引用，已提示使用 view_image")
                self.session.log_dict_info({
                    "role": "system",
                    "content": f"[视觉提示] 文档内嵌媒体引用: {media_paths}"
                })
        except Exception as e:
            logger.warning(f"需求文档预读失败: {path} ({e})")
            self.api_messages.append_micro_info(
                "system", f"[预读失败] {path}（{e}），请用 file_view 读取"
            )

    def _detect_embedded_media_refs(self, doc_content: str) -> list:
        """检测需求文档内容中内嵌的媒体文件引用（F4，复用 L0 正则检测）。

        场景：用户输入只引用了需求文档（如 "do as xxx.txt"），但文档内部
        嵌套引用了图片——L0/L1 分流只看用户输入层，覆盖不到这类嵌套引用。
        预读注入后对文档正文做一次媒体扩展名扫描，发现引用即注入视觉提示，
        引导 LLM 用 view_image 阅读图片而非 file_view 硬读。

        返回去重后的媒体路径清单。仅做扩展名匹配，不做存在性检查
        （路径问题由 view_image 执行时暴露）。
        """
        try:
            from wdclaude.vision.input_resolver import resolve_input
            media_exts = self._get_media_extensions()
            if not media_exts or not doc_content:
                return []
            norm_exts = set()
            for e in media_exts:
                e = str(e).strip().lower()
                if e:
                    norm_exts.add(e if e.startswith(".") else "." + e)
            resolved = resolve_input(doc_content, media_exts)
            seen = set()
            media_paths = []
            for ref in resolved.references:
                fname = ref.path.rsplit("/", 1)[-1].lower()
                if any(fname.endswith(e) for e in norm_exts) and ref.path not in seen:
                    seen.add(ref.path)
                    media_paths.append(ref.path)
            return media_paths
        except Exception as e:
            logger.warning(f"文档内嵌媒体引用检测失败: {e}")
            return []

    @staticmethod
    def _references_file(text: str) -> bool:
        """检测用户输入是否引用了文件路径。

        匹配常见模式：
        - 盘符路径：D:\\xxx 或 D:/xxx
        - 相对路径引用：xxx.txt, xxx.md, .py 等
        - 包含"文档"、"文件"、"spec"等关键词
        """
        #  import re
        # 盘符路径
        if re.search(r'[A-Za-z]:[/\\]', text):
            return True
        # 常见文档扩展名引用
        if re.search(r'\.(txt|md|py|json|yaml|yml|xml|csv|html|js|ts)\b', text, re.IGNORECASE):
            return True
        return False

    # _on_llm_req，表示在发请求信息给LLM之前，做的（部分）事情，可能不是所有事情
    def _on_llm_req(self, turn, user_input):
        # 先开始新的 Turn 缓冲（必须在 log_dict_info 之前调用，
        # 否则 log_turn 会重置 _turn_buffer，丢弃记忆召回等内容）
        self.session.log_turn(turn)

        # 消费上一 Turn 延迟存储的召回日志（file_view 后触发的召回）
        # 推迟到当前新 Turn 记录，避免在触发召回的 Turn 中产生重复的"记忆召回"小节
        if self._pending_retrieval_result is not None:
            self.session.log_memory_retrieval(self._pending_retrieval_result)
            self._pending_retrieval_result = None

        # 第一轮，需要初始化（init_session 内部有守护，只执行一次）
        if turn == 1:
            self.session.init_session()

            # 斜杠命令模式：api_messages 已由 reset_with_command 设置好，跳过 init_api_msg
            if self._command_context:
                # 用命令名+参数作为记忆检索的查询文本
                query_text = self._command_context.get("user_argument") or user_input
            else:
                self.api_messages.init_api_msg(user_input)
                query_text = user_input

            # ===== L0/L1 输入分流（multimodal_spec 3.1，静默执行，不预告）=====
            # L0 检测 -> L1 角色裁定 -> B 提取/预读清单/召回查询增强；
            # 零候选时零侵入（recall_query 即原文）；fail-fast 时置 _input_route_fatal
            recall_query, preread_paths, b_message = self._route_input_references(query_text)

            # 输入分流 fail-fast 兜底检查（防御性保留：当前正常流程不会置位）
            if self._input_route_fatal:
                # 孤儿消息清理：本轮 A 从未发给 LLM，残留会使下一轮出现
                # 无回应的 user 消息。非命令模式下回滚刚注入的 A
                # （命令模式的消息生命周期由 reset_with_command 管理，不在此处理）
                msgs = self.api_messages.api_messages
                if msgs and msgs[-1].get("role") == "user" and not self._command_context:
                    msgs.pop()
                return datetime.now().strftime("%Y-%m-%d %H : %M : %S")

            # B 消息注入（B 前 A 后，4.7）：先插会话级视觉说明（4.6，仅一次）
            # 再插 B，两次 insert 均落在 A 之前，得到 [说明, B, A] 排布
            if b_message:
                if not self._vision_notice_injected:
                    self._vision_notice_injected = True
                    from wdclaude.vision.vision_extractor import SESSION_VISION_NOTICE
                    self.api_messages.insert_before_last_user(SESSION_VISION_NOTICE)
                self.api_messages.insert_before_last_user(b_message)
                self.session.log_dict_info({
                    "role": "system",
                    "content": "[视觉提取] B 消息已注入（" + b_message[:80].replace(chr(10), " ") + "...）"
                })

            # 需求文档预读注入（spec 3.1：召回查询不含预读内容——recall_query
            # 已在分流时构建完毕；operation_target 不预读，杜绝"提到文件就预读"）
            for preread_path in preread_paths:
                self._inject_preread(preread_path)

            # 注入记忆上下文（通过 MemoryInterface 统一接口）
            # 记忆召回时机（multimodal_spec 3.1）：
            # - 零引用 / 全部 operation_target -> 立即召回（静默分类后直接打印结果）
            # - requirement_doc -> 召回查询 = A + summary + 引用角色清单（预读不参与）
            # - 回退模式（_pending_memory_recall=True，vision 关闭或装配失败）
            #   -> 保持旧行为，延迟到 file_view 后召回
            if self._pending_memory_recall:
                self._memory_used = True
            else:
                try:
                    # 传入当前 session_id，确保不召回本 session 产生的记忆
                    current_session_id = getattr(self.session, 'session_file_name', '')
                    # 设置 token 统计上下文，标记为记忆召回（查询为分流增强后的 recall_query）
                    chat_llm.set_context(query=recall_query, turn=f"turn{turn}-recall")
                    mem_context = self._memory.get_context_for_query(
                        recall_query, exclude_session_id=current_session_id
                    )
                    chat_llm.set_context()  # 清除上下文
                    self._memory_used = True

                    # 获取 RetrievalResult 用于日志和 CLI 显示
                    retrieval_result = None
                    if hasattr(self._memory, 'get_last_retrieval_result'):
                        retrieval_result = self._memory.get_last_retrieval_result()

                    # CLI 打印：策略 + 最终结果（简洁格式）
                    if retrieval_result:
                        try:
                            from wdclaude.cli import cli_print
                            cli_print.print_memory_recall(retrieval_result)
                        except Exception as e:
                            logger.warning(f"CLI打印记忆召回失败: {e}")

                    # 日志记录：完整召回过程（含各阶段）
                    if retrieval_result:
                        self.session.log_memory_retrieval(retrieval_result)

                    # 解析检索结果数量
                    recall_count = self._count_recalled(mem_context) if mem_context else 0
                    if not retrieval_result:
                        self._print_info(f"[记忆召回] 已召回 {recall_count} 条相关记忆")

                    if mem_context:
                        self.api_messages.append_micro_info("system", mem_context)
                        logger.debug(f"记忆上下文已注入，长度: {len(mem_context)}（查询: {recall_query[:120]}）")
                    elif not retrieval_result:
                        # retrieval_result 已通过 log_memory_retrieval 记录，不重复注入
                        self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n没有召回到相关记忆\n[召回数: 0]")
                except Exception as e:
                    chat_llm.set_context()  # 确保异常时也清除上下文
                    logger.warning(f"记忆上下文注入失败: {e}")
                    self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n记忆召回异常，跳过注入\n[召回数: 0]")

        # 回退：如果延迟召回仍未触发（LLM 未调用 file_view），用原始用户输入执行召回
        if turn > 1 and self._pending_memory_recall:
            self._pending_memory_recall = False
            try:
                current_session_id = getattr(self.session, 'session_file_name', '')
                chat_llm.set_context(query=self._current_user_input, turn=f"turn{turn}-recall-fallback")
                mem_context = self._memory.get_context_for_query(
                    self._current_user_input, exclude_session_id=current_session_id
                )
                chat_llm.set_context()

                # 获取 RetrievalResult 用于日志和 CLI 显示
                retrieval_result = None
                if hasattr(self._memory, 'get_last_retrieval_result'):
                    retrieval_result = self._memory.get_last_retrieval_result()
                if retrieval_result:
                    try:
                        from wdclaude.cli import cli_print
                        cli_print.print_memory_recall(retrieval_result)
                    except Exception as e:
                        logger.warning(f"CLI打印回退记忆召回失败: {e}")
                # 延迟到下一 Turn 记录召回日志，避免在当前 Turn 产生重复的"记忆召回"小节
                if retrieval_result:
                    self._pending_retrieval_result = retrieval_result

                recall_count = self._count_recalled(mem_context) if mem_context else 0
                if not retrieval_result:
                    self._print_info(f"[记忆召回] 已召回 {recall_count} 条相关记忆（回退到用户输入）")
                if mem_context:
                    self.api_messages.append_micro_info("system", mem_context)
                    logger.debug(f"回退记忆上下文已注入，长度: {len(mem_context)}")
                elif not retrieval_result:
                    # retrieval_result 已通过 log_memory_retrieval 记录，不重复注入
                    self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n没有召回到相关记忆\n[召回数: 0]")
            except Exception as e:
                chat_llm.set_context()
                logger.warning(f"回退记忆上下文注入失败: {e}")
                self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n记忆召回异常，跳过注入\n[召回数: 0]")

        # 倒数最后一轮，命令式提醒
        if turn == self.max_turns and self.is_multi_turns:
            command = "命令：如果你已完成所有修改，请立即调用 <llm_tool>done</llm_tool> 结束任务。不要继续调用其他工具。"
            self.api_messages.append_micro_info("system", command)

        # 每轮注入 [TODO_STATUS] 上下文（如果有活跃的 todo 列表）
        todo_context = self._todo_manager.get_context_message()
        if todo_context:
            self.api_messages.append_micro_info("system", todo_context)
            self.session.log_dict_info({"role": "system", "content": todo_context})

        # 事前记录发送给LLM的req（log_turn 已在方法开头调用）
        self.session.log_llm_req(self.api_messages.get_msg())

        # 这里的thinking，不是LLM的thinking，是LLM的整个应答
        thinking_begin = datetime.now().strftime("%Y-%m-%d %H : %M : %S")

        return thinking_begin


    def _on_llm_rsp(self, turn, thinking_begin, ai_response, reasoning_content):
        # LLM回答结束的时间戳
        thinking_end = datetime.now().strftime("%Y-%m-%d %H : %M : %S")

        # 记录推理内容（如果提供商支持）
        self.session.log_reasoning_content(reasoning_content)

        # 去除thinking部分（针对 Claude 风格，DeepSeek 无此部分）
        ai_response_clean = strip_thinking(ai_response)

        # 如果实际响应为空但推理内容存在，使用推理内容作为有效响应
        # （某些 LLM 将所有内容放在思考过程中，实际响应为空）
        if not ai_response_clean.strip() and reasoning_content:
            ai_response_clean = reasoning_content.strip()

        # 压缩工具标签用于 MD 日志展示（api_messages 使用原始版本）
        md_content = self._compress_assistant_message(ai_response_clean)

        # 记录LLM回应（日志保留完整内容，md_content 为压缩后的展示版本）
        self.session.log_llm_rsp(ai_response, md_content=md_content)

        # 直接附上原始的LLM的response（去除thinking部分），不压缩
        self.api_messages.append_llm_response(ai_response_clean)

        # 是否给用户显示LLM的think过程
        if not self.show_thinking:
            ai_response_show = ai_response_clean
        else:
            ai_response_show = ai_response

        # 仅从 ai_response 解析工具（不 fallback 到 reasoning_content）
        remaining_text, tools = tool_executor.parse_tools(ai_response_show)

        # 打印部分 LLM response（有些内容不打印，显示一分神秘感）
        self._print_info(f"Thinking-{turn}, 开始时间：{thinking_begin}")

        # 打印推理内容给前端（打字机效果，完成后自动折叠）
        if reasoning_content:
            if self._print_llm_reasoning:
                self._print_llm_reasoning(reasoning_content, turn)
            else:
                self._print_llm_rsp(reasoning_content)

        self._print_info(f"Thinking-{turn}, 结束时间：{thinking_end}")

        if remaining_text:
            self._print_llm_rsp(remaining_text)

        # 记忆存储已移至 run() 中的 _save_turn_memory()，统一打包整轮对话

        return tools, remaining_text


    def _get_follow_up_prompt(self, retry_count: int) -> str:
        """根据追问次数返回递进式的提醒消息，给 LLM 两条路：输出工具或输出 done。

        Args:
            retry_count: 追问次数（1, 2, 3）

        Returns:
            对应次数的追问消息文本
        """
        prompts = {
            1: "[系统提醒] 你既没有输出工具，也没有输出 <done>。如果任务尚未完成，请继续输出下一个工具。如果你在等待用户的回应，请输出 <done> 结束本轮。",
            2: "[系统追问] 请立即输出工具继续执行，或输出 <done> 结束本轮。",
            3: "[最终提醒] 这是你最后的机会。输出工具继续，或输出 <done> 结束。否则本轮将自动结束，等待用户输入。",
        }
        return prompts.get(retry_count, prompts[3])

    def _get_format_error_prompt(self, retry_count: int) -> str:
        """格式纠错追问：检测到标签形态但未解析出任何工具时（402 现场）。

        典型现场：LLM 输出 '<web_search max_results="5" query="深圳今天天气.../web_search>'
        ——属性值引号未闭合 + 自闭合标签误用容器闭合。strict/loose/lenient 均无法
        解析时，标签文本被当作正文直接展示给用户，用户只看到裸 XML，工具从未执行。

        Args:
            retry_count: 追问次数（1, 2, 3）

        Returns:
            携带正确格式示例的纠错追问消息
        """
        prompts = {
            1: (
                "[格式错误] 你上一条回复中的工具标签格式错误，未能解析执行。常见错误：属性值引号未闭合、"
                "自闭合标签误用了容器式闭合标签。正确格式示例：\n"
                '<file_view path="D:/AI/MyClaude/spec/xxx.md"/>\n'
                '<web_search query="搜索关键词" max_results="5"/>\n'
                "请严格按照上述格式重新输出工具调用（自闭合标签以 /> 结尾，属性值必须有成对双引号）。"
            ),
            2: (
                "[格式错误-重复] 工具标签仍然无法解析。请检查：1) 每个属性值都有成对双引号；"
                "2) 自闭合标签（file_view/web_search/use_skill 等）必须以 /> 结尾，不要写 </tag>；"
                "3) 不要在属性值中换行。重新输出正确的工具调用。"
            ),
            3: (
                "[最后机会] 请输出一个格式正确的工具调用，或直接输出 <done> 结束本轮。"
                "工具调用格式：工具名 + 属性（值用双引号包裹）+ 以 /> 结尾，"
                '例如 <web_search query="深圳天气"/>'
            ),
        }
        return prompts.get(retry_count, prompts[3])

    def _log_aborted_intent_metrics(self, remaining_text: str) -> None:
        """层 3 埋点：记录首轮纯文本的流产意图特征指标（仅观测，不参与判定）。

        检测结果不作为任何分支的判定依据（判定已由层 1 协议收敛统一承担），
        仅用于：
        1. 统计流产发生率，验证层 1 修复前后的效果对比
        2. 为后续是否引入 LLM 语义判定（原方案层 2）积累实证样本

        观测指标：
        - colon_ending: 正文以全角/半角冒号结尾（终局性回答几乎不会这样收尾）
        - intent_phrase: 正文含 先看/接下来/继续看/让我/看一下/去验证 等过渡语
        """
        if not remaining_text or not remaining_text.strip():
            return
        text = remaining_text.strip()
        signals = []
        if text.endswith("：") or text.endswith(":"):
            signals.append("colon_ending")
        intent_phrases = ("先看", "接下来", "继续看", "让我", "看一下", "去验证")
        if any(p in text for p in intent_phrases):
            signals.append("intent_phrase")
        if signals:
            tail = text[-30:].replace("\n", " ")
            self.session.log_dict_info({
                "role": "system",
                "content": f"[观测埋点] 首轮纯文本流产意图特征: {', '.join(signals)} | 尾部: ...{tail}"
            })

    def _follow_up_for_tools(self, format_error: bool = False):
        """无工具时追问 LLM 最多 3 次，尝试获取工具列表。

        追问成功后，将追问消息和 LLM 回复追加到正式 api_messages，
        确保后续工具执行结果前面有完整的 assistant 消息，避免上下文断裂。

        Args:
            format_error: 是否为格式纠错模式（检测到标签形态但解析失败），
                          使用携带正确格式示例的纠错提示词

        Returns:
            解析到的工具列表；若 3 次追问仍无工具，返回空列表。
        """
        #  import copy

        if self._no_tool_retry >= 3:
            return []

        self._no_tool_retry += 1
        if format_error:
            prompt = self._get_format_error_prompt(self._no_tool_retry)
        else:
            prompt = self._get_follow_up_prompt(self._no_tool_retry)
        # 方案 E：第一次追问静默（不打印到屏幕），只注入给 LLM
        # 格式纠错场景不静默：让用户看到发生了什么
        silent = (self._no_tool_retry == 1) and not format_error
        if not silent:
            self._print_info(f"[追问 {self._no_tool_retry}/3] {prompt}")

        try:
            # 构建临时消息列表（深拷贝，不污染正式对话历史）
            temp_msgs = copy.deepcopy(self.api_messages.get_msg())
            temp_msgs.append({"role": "user", "content": prompt})

            ai_response, is_truncated, reasoning_content, usage = chat_llm.chat_with_retry(temp_msgs)

            # ===== Prompt 超长熔断（415 现场）=====
            # 追问请求同样超长时，继续追问必然失败（415 现场：追问 3 次全部
            # 返回同样的超长错误），直接放弃追问，避免无谓的调用与等待
            if ai_response.startswith("[API_ERROR:") and any(
                kw in ai_response.lower()
                for kw in ("max length", "context length", "context_length",
                           "too long", "too large", "prompt is too")
            ):
                self._print_info("[上下文超限] 追问请求同样超过模型长度限制，放弃追问，结束本轮")
                self.session.log_dict_info({
                    "role": "system",
                    "content": "[上下文超限熔断] 追问请求 Prompt 超长，放弃追问"
                })
                self._no_tool_retry = 3  # 置满计数器，阻止外层继续追问
                return []

            # 累加追问的 token 统计到内存计数器并持久化
            if usage:
                cached = usage.get("cached_tokens", 0)
                prompt_total = usage.get("prompt_tokens", 0)
                self.prompt_cache_hit += cached
                self.prompt_cache_miss += (prompt_total - cached)
                self.completion_tokens += usage.get("completion_tokens", 0)

                query_text = getattr(self, '_current_query_text', '')
                record_token_usage(
                    model_name=chat_llm.model_name,
                    prompt_tokens=prompt_total,
                    cached_tokens=cached,
                    completion_tokens=usage.get("completion_tokens", 0),
                    query=query_text,
                    turn=f"turn-followup{self._no_tool_retry}",
                )

            # 记录追问对话到 session（用于日志审计）
            reason = f"[系统追问] Turn 中未检测到工具调用，发起第 {self._no_tool_retry}/3 次追问，尝试获取工具或 done 信号"
            self.session.log_dict_info({"role": "system", "content": reason})
            self.session.log_turn(-self._no_tool_retry)  # 负数 turn 表示追问
            self.session.log_llm_req(temp_msgs)
            md_content = self._compress_assistant_message(strip_thinking(ai_response))
            self.session.log_llm_rsp(ai_response, md_content=md_content)
            self.session.log_reasoning_content(reasoning_content)
        except Exception as e:
            logger.error(f"追问 LLM 失败: {e}")
            return []

        # 解析工具（仅从 ai_response，不回退到 reasoning_content）
        ai_response_clean = strip_thinking(ai_response)

        # 如果实际响应为空但推理内容存在，使用推理内容作为有效响应
        if not ai_response_clean.strip() and reasoning_content:
            ai_response_clean = reasoning_content.strip()

        remaining_text, tools = tool_executor.parse_tools(ai_response_clean)

        # ===== 回声防护（408 现场）=====
        # LLM 逐字复述追问提示词时（提示词含 <done> 字面示例），parse_tools
        # 会把复述文本中的 <done> 误提取为幻影 done 工具：正文被挖出空洞、
        # done 消息显示为残句、任务被误判完成。检测到复述则视为无效回复，
        # 忽略其中的幻影工具并继续追问。
        if tools and prompt:
            probe = re.sub(r'^\[[^\]]+\]\s*', '', prompt)[:15].strip()
            if probe and probe in ai_response_clean:
                tools = []
                remaining_text = ""
                self._print_info("[回声检测] 追问回复复述了提示词文本，已忽略其中的幻影工具")
                self.session.log_dict_info({
                    "role": "system",
                    "content": "[回声检测] 追问回复复述了提示词文本（含工具标签示例），判定为无效回复，继续追问"
                })

        # 打印追问回复中的正文（此前正文被丢弃，导致用户看不到 LLM 的说明文字，
        # 出现"done 消息声称已提供内容但用户从未看到"的现象）
        if remaining_text:
            self._print_llm_rsp(remaining_text)

        if tools:
            if not silent:
                self._print_info(f"[追问结果] 成功获取到 {len(tools)} 个工具，进入执行")
            # 追问成功：将追问消息和 LLM 回复追加到正式 api_messages，
            # 确保后续工具执行结果前面有完整的 assistant 消息，避免上下文断裂
            self.api_messages.append_micro_info("system", prompt)
            self.api_messages.append_llm_response(ai_response_clean)
        else:
            if not silent:
                self._print_info(f"[追问结果] 第 {self._no_tool_retry}/3 次仍未获得工具")

        return tools


    def _handle_tools(self, tools, remaining_text: str = ""):
        """执行工具并返回 (quit_chat, tool_exec_info)。
        tool_exec_info 为列表，每个元素是 {"tool": 工具名, "params": 参数, "result": 结果文本}。

        Args:
            tools: 解析出的工具列表
            remaining_text: 工具标签剥离后的正文（用于标签形态检测）
        """
        # 1. 如果 LLM response 中没有工具
        if not tools:
            # ===== 402 修复：第一轮无工具但残留标签形态 → 格式纠错追问 =====
            # LLM 输出畸形标签（引号未闭合等）时，标签文本被当正文展示，
            # 工具从未执行。检测到标签形态时优先追问纠错，而非静默退出单轮。
            tag_shape = tool_executor.has_tool_tag_shape(remaining_text)
            if self.is_multi_turns is None:
                if tag_shape:
                    # 有标签形态但未解析出工具：格式错误，触发纠错追问
                    self.is_multi_turns = True
                    self._print_info("[格式异常] 检测到工具标签形态但解析失败，发起格式纠错追问")
                    self.session.log_dict_info({
                        "role": "system",
                        "content": "[格式异常] 工具标签格式错误（引号未闭合/闭合标签误用），触发格式纠错追问"
                    })
                    while self._no_tool_retry < 3:
                        tools = self._follow_up_for_tools(format_error=True)
                        if tools:
                            self._no_tool_retry = 0
                            break
                    if not tools:
                        self._print_info("格式纠错追问 3 次后仍未获得工具，本轮结束，等待用户输入")
                        self.session.log_dict_info({"role": "system", "content": "格式纠错追问 3 次后仍未获得工具，本轮结束，等待用户输入"})
                        return ChatOrNot.QuitByNoneTool, []
                else:
                    # ===== 层 1 协议收敛：首轮纯文本无 done → 静默追问一次 =====
                    # 协议规定 done 是唯一合法终局信号（含闲聊场景，见系统提示词 Layer 5）。
                    # 首轮纯文本无工具无 done 属于违规输出（典型现场：LLM 说了
                    # "先看一下..."就停住，以为还有下一轮——两次停住事故的根因）。
                    # 处置：静默追问一次给 LLM 补救机会；仍无果则按单轮正常退出，
                    # 兜底行为与旧版完全一致（零回归）。
                    # 注：_follow_up_for_tools 首次追问默认静默（不打印屏幕），用户无感
                    self._log_aborted_intent_metrics(remaining_text)  # 层 3 埋点：仅观测，不参与判定

                    tools = self._follow_up_for_tools()
                    if tools:
                        # 追问成功：LLM 补出了工具或 done，继续正常流程。
                        # 将追问计数置满以跳过下方"后续轮次"追问循环，避免重复追问；
                        # 流出 if not tools 块后 "# 有工具时重置追问计数器" 会将其归零
                        self._no_tool_retry = 3
                        self.session.log_dict_info({
                            "role": "system",
                            "content": "[协议收敛] 首轮纯文本追问成功，LLM 已补出终局信号或工具"
                        })
                    else:
                        # 追问无果：按单轮正常退出（兜底路径，与旧行为一致）
                        self.is_multi_turns = False
                        self.session.log_dict_info({
                            "role": "system",
                            "content": "LLM 未调用工具，首轮协议收敛追问一次后仍无工具/done，按单轮结束，等待用户输入"
                        })
                        return ChatOrNot.QuitByNoneTool, []

            # is_multi_turns = True 的后续轮次：无工具无 done → 触发追问
            while self._no_tool_retry < 3:
                tools = self._follow_up_for_tools(format_error=tag_shape)
                if tools:
                    # 追问得到了工具，重置计数器
                    self._no_tool_retry = 0
                    break
                # 未获得工具，_follow_up_for_tools 已自增 _no_tool_retry，继续循环追问
            if not tools:
                # 追问无果，兜底结束 Turn
                self._print_info("追问 3 次后仍未获得工具，本轮结束，等待用户输入")
                self.session.log_dict_info({"role": "system", "content": "追问 3 次后仍未获得工具，本轮结束，等待用户输入"})
                return ChatOrNot.QuitByNoneTool, []
            # 继续往下执行（会进入后面的 exec_tools / done_tools 处理）

        # 有工具时重置追问计数器
        self._no_tool_retry = 0

        # 第一轮有工具 → 动态确定为多轮场景
        if self.is_multi_turns is None:
            self.is_multi_turns = True

        # 既然有工具，那就执行工具'''
        done_tools = [t for t in tools if t["llm_tool"] == "done"]
        exec_tools = [t for t in tools if t["llm_tool"] != "done"]

        # ===== AskUserQuestion 特殊处理 =====
        has_ask_user = any(t["llm_tool"] == "AskUserQuestion" for t in exec_tools)

        # TodoWrite 排到最前面优先执行（先更新计划，再执行动作）
        # 排序键：todowrite=0, AskUserQuestion=1, 其他=2
        exec_tools.sort(key=lambda t: 0 if t["llm_tool"] == "todowrite" else (1 if t["llm_tool"] == "AskUserQuestion" else 2))

        # ===== 死循环熔断（通用版，415 现场）=====
        # 架构契约：QueryLoop 维护 last_tool_sig，连续两轮工具签名完全一致时强制终止。
        # 旧版仅对 AskUserQuestion 生效；415 现场 LLM 连续多轮对同一文件调用
        # 完全相同的 file_view（limit=160），每轮结果都 append 进 api_messages，
        # 上下文无限膨胀直至 Prompt exceeds max length。
        # 签名 = 各工具名 + 关键短参数（path/command/query/question/name/intent/
        # sheet/limit/offset/max_results）+ 大内容字段哈希（old/new/content 取
        # md5 前 8 位）。file_view 的 limit/offset 参与签名，分页读取（offset
        # 递增）不算重复；str_replace 修正内容后哈希变化，合法重试不被误熔断。
        def _sig_of(t):
            p = t.get("params", {}) or {}
            bits = []
            for k in ("path", "command", "query", "question", "name", "intent", "sheet", "url"):
                v = p.get(k)
                if v is not None:
                    bits.append(f"{k}={v}")
            for k in ("limit", "offset", "max_results"):
                v = p.get(k)
                if v is not None:
                    bits.append(f"{k}={v}")
            # 大内容字段以内容哈希参与签名：仅当内容也完全一致时才判定重复，
            # 避免 LLM 修正 old/new 后对同一文件的合法重试被误熔断
            for k in ("old", "new", "content"):
                v = p.get(k)
                if isinstance(v, str) and v:
                    digest = hashlib.md5(v.encode("utf-8")).hexdigest()[:8]
                    bits.append(f"{k}#{digest}")
            return f"{t['llm_tool']}({','.join(bits)})"

        current_sig = "|".join(_sig_of(t) for t in exec_tools) if exec_tools else None
        if current_sig is not None and current_sig == self._last_tool_sig:
            self._print_info("[熔断] 检测到连续两轮调用完全相同的工具列表，强制终止以避免死循环")
            self.session.log_dict_info({
                "role": "system",
                "content": f"[熔断] 连续两轮工具签名完全一致（{current_sig[:200]}），强制终止"
            })
            return ChatOrNot.QuitByNoneTool, []
        self._last_tool_sig = current_sig

        tool_exec_info = []
        fv_seen = set()  # 本轮已执行的 file_view 键（path/limit/offset），用于单轮去重
        fv_dup_warned = 0  # 单轮内重复 file_view 的阻断提示计数（超过 3 个后静默跳过）
        # 本轮是否有工具真正执行（未被前置防护阻断）。
        # 用于 Tool-Done 同轮分离判定：只有工具真的跑过，才值得丢弃 done
        # 让结果回灌；若全部被阻断，done 照常生效，避免任何潜在空转。
        any_effective_tool = False

        # 执行普通工具
        if exec_tools:
            self._print_info(f"执行 {len(exec_tools)} 个工具...")

            for t in exec_tools:
                # ===== 安全点（优雅退出阶段二）：工具执行前轮询中断标志 =====
                # 工具（尤其 bash/测试）可能长时间运行，执行前检查可避免
                # 中断请求已发生后仍继续启动新的长耗时操作
                if shutdown_mgr.is_interrupted():
                    self._print_info("[优雅退出] 收到退出请求，剩余工具跳过执行")
                    self.session.log_dict_info({
                        "role": "system",
                        "content": f"[优雅退出] 用户中断（{shutdown_mgr.interrupt_reason}），剩余工具跳过执行"
                    })
                    return ChatOrNot.QuitByNoneTool, tool_exec_info

                self._print_tool_call(t["llm_tool"], t["params"])  # 打印：工具名称，工具参数
                self.session.log_tool_call(t["llm_tool"], t["params"])

                blocked = False  # 工具是否被防护逻辑阻断（file_view 重复防护等）

                # ===== TodoWrite 拦截：交给 TodoManager 处理 =====
                if t["llm_tool"] == "todowrite":
                    try:
                        result_msg = self._todo_manager.update_from_xml(
                            t["params"].get("content", "")
                        )
                        # 通知 CLI 渲染 todo 列表
                        if self._on_todo_update:
                            self._on_todo_update(self._todo_manager.get_display_data())
                    except Exception as e:
                        logger.error(f"TodoWrite 执行异常: {e}")
                        result_msg = {
                            "role": "user",
                            "content": f"[ERROR] TodoWrite 执行失败: {e}"
                        }
                else:
                    # ===== file_view 重复防护（415 现场）=====
                    # 两层防护，杜绝同一文件被无限重复读取撑爆上下文：
                    # 1) 单轮去重：同一条回复中重复的 file_view（相同 path/limit/
                    #    offset）只执行第一个，其余前 3 个生成阻断提示、之后静默
                    #    跳过——防止 LLM 单轮输出成百上千个重复标签撑爆上下文
                    # 2) 跨轮硬阻断：同一文件累计查看第 3 次注入最终警告（仍执行），
                    #    第 4 次起直接拒绝执行。旧版仅注入软提醒且可被 LLM 无视
                    #    （415 现场：同一文件被连续读取 2600+ 次，直至 Prompt 超限）
                    if t["llm_tool"] == "file_view":
                        fv_path = t["params"].get("path", "")
                        fv_key = (fv_path, t["params"].get("limit"), t["params"].get("offset"))
                        if fv_key in fv_seen:
                            # 单轮内重复：前 3 个生成阻断提示告知 LLM，之后静默跳过
                            # （防止极端场景下单轮海量重复标签的 BLOCKED 消息本身撑大上下文）
                            blocked = True
                            if fv_dup_warned < 3:
                                fv_dup_warned += 1
                                result_msg = {
                                    "role": "user",
                                    "content": f"[BLOCKED] 本轮内重复的 file_view 已跳过: {fv_path}"
                                }
                            else:
                                self.session.log_dict_info({
                                    "role": "system",
                                    "content": f"[熔断] 单轮内重复 file_view 已静默跳过: {fv_path}"
                                })
                                continue
                        else:
                            fv_seen.add(fv_key)
                            if fv_path:
                                self._file_view_counts[fv_path] = self._file_view_counts.get(fv_path, 0) + 1
                                count = self._file_view_counts[fv_path]
                                if count == 3:
                                    reminder = (
                                        f"[系统提醒] 文件 {fv_path} 已被查看 {count} 次。"
                                        f"这是最后一次允许的查看，下次重复查看将被系统直接拒绝。"
                                        f"文件内容已在上下文中，请尽快基于已读内容修改或结束任务。"
                                    )
                                    self.api_messages.append_micro_info("system", reminder)
                                    self._print_info(f"[file_view 提醒] {fv_path} 已查看 {count} 次，下次将拒绝执行")
                                elif count > 3:
                                    blocked = True
                                    result_msg = {
                                        "role": "user",
                                        "content": (
                                            f"[BLOCKED] 文件 {fv_path} 本次任务中已查看 {count} 次，本次拒绝执行。"
                                            f"内容已在上下文中，请基于已读内容修改（str_replace）或输出 <done> 结束任务。"
                                        )
                                    }
                                    self._print_info(f"[file_view 熔断] {fv_path} 已查看 {count} 次，本次拒绝执行")
                                    self.session.log_dict_info({
                                        "role": "system",
                                        "content": f"[熔断] file_view {fv_path} 重复查看 {count} 次，本次拒绝执行"
                                    })
                    if not blocked:
                        try:
                            result_msg = tool_executor.execute_code_tool(t)  # 工具执行
                        except Exception as e:
                            # 视觉链路 fail-fast（multimodal_spec D8）：view_image 的
                            # VisionFatalError 终止本轮，不降级为普通错误文本
                            try:
                                from wdclaude.vision.vision_client import VisionFatalError
                                if isinstance(e, VisionFatalError):
                                    self._input_route_fatal = True
                                    self._print_info(f"[视觉终止] {e}；建议检查网络/换图/减少图片数量后重试")
                                    self.session.log_dict_info({"role": "system", "content": f"[视觉-fail-fast] {e}"})
                                    return ChatOrNot.QuitByNoneTool, tool_exec_info
                            except ImportError:
                                pass
                            logger.error(f"工具执行异常 [{t['llm_tool']}]: {e}")
                            result_msg = {
                                "role": "user",
                                "content": f"[ERROR] 工具 {t['llm_tool']} 执行失败: {e}"
                            }

                if not blocked:
                    any_effective_tool = True

                self._print_tool_result(t["llm_tool"], result_msg.get("content", ""), t["params"])  # 打印：工具执行结果
                self.session.log_tool_result(t["llm_tool"], result_msg)

                # TodoWrite 执行后记录 todo 快照到会话日志
                if t["llm_tool"] == "todowrite":
                    self.session.log_todo_snapshot(self._todo_manager.get_display_data())

                # 将 tool 的执行结果，append 到 api_messages
                self.api_messages.append_tool_exec_result(result_msg)

                # （file_view 重复查看检测已前移至工具执行前，见上方 blocked 防护逻辑）

                # ===== 延迟记忆召回：file_view 执行后用文件内容触发召回 =====
                # （被阻断/拒绝的 file_view 不触发：其 result 是 BLOCKED 提示而非文件内容）
                if t["llm_tool"] == "file_view" and not blocked and self._pending_memory_recall:
                    self._pending_memory_recall = False
                    try:
                        file_content = result_msg.get("content", "")
                        # 用文件内容 + 用户原始输入组合作为查询文本
                        # 不截断，由 memory_ex 的意图压缩统一处理
                        recall_query = f"{self._current_user_input}\n{file_content}"
                        current_session_id = getattr(self.session, 'session_file_name', '')
                        chat_llm.set_context(query=self._current_user_input, turn=f"turn-recall-fileview")
                        mem_context = self._memory.get_context_for_query(
                            recall_query, exclude_session_id=current_session_id
                        )
                        chat_llm.set_context()

                        # 获取 RetrievalResult 用于日志和 CLI 显示
                        retrieval_result = None
                        if hasattr(self._memory, 'get_last_retrieval_result'):
                            retrieval_result = self._memory.get_last_retrieval_result()
                        if retrieval_result:
                            try:
                                from wdclaude.cli import cli_print
                                cli_print.print_memory_recall(retrieval_result)
                            except Exception as e:
                                logger.warning(f"CLI打印延迟记忆召回失败: {e}")
                        # 延迟到下一 Turn 记录召回日志，避免在当前 Turn 产生重复的"记忆召回"小节
                        if retrieval_result:
                            self._pending_retrieval_result = retrieval_result

                        recall_count = self._count_recalled(mem_context) if mem_context else 0
                        if not retrieval_result:
                            self._print_info(f"[记忆召回] 已召回 {recall_count} 条相关记忆（基于文件内容）")
                        if mem_context:
                            self.api_messages.append_micro_info("system", mem_context)
                            logger.debug(f"延迟记忆上下文已注入，长度: {len(mem_context)}")
                        elif not retrieval_result:
                            # retrieval_result 已通过 log_memory_retrieval 记录，不重复注入
                            self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n没有召回到相关记忆\n[召回数: 0]")
                    except Exception as e:
                        chat_llm.set_context()
                        logger.warning(f"延迟记忆召回失败: {e}")
                        self.api_messages.append_micro_info("system", "[MEMORY_INJECTION_v1]\n记忆召回异常，跳过注入\n[召回数: 0]")

                # 收集工具执行信息（用于记忆存储）
                tool_exec_info.append({
                    "tool": t["llm_tool"],
                    "params": t["params"],
                    "result": result_msg.get("content", "")[:500],  # 截断过长结果
                })

        # ===== AskUserQuestion 后跳过 done 检测 =====
        # 如果本轮包含 AskUserQuestion，忽略同轮的 done 信号，强制继续下一轮
        if has_ask_user:
            return ChatOrNot.Continue, tool_exec_info

        # ===== 死循环熔断：纯 todowrite 轮的状态变化检测 =====
        non_todo_tools = [t for t in exec_tools if t["llm_tool"] != "todowrite"]
        has_todowrite = any(t["llm_tool"] == "todowrite" for t in exec_tools)

        if has_todowrite and not non_todo_tools:
            # 纯 todowrite 轮：检查 todo 状态是否有变化
            current_todo_sig = self._todo_manager.todo_list.status_signature()
            if self._last_todo_sig is not None and self._last_todo_sig == current_todo_sig:
                self._print_info("[熔断] 连续两轮纯 todowrite 且状态无变化，强制终止")
                self.session.log_dict_info({
                    "role": "system",
                    "content": "[熔断] TodoWrite 连续两轮状态无变化，强制终止"
                })
                return ChatOrNot.QuitByNoneTool, tool_exec_info
            self._last_todo_sig = current_todo_sig
        else:
            # 有非 todowrite 工具，重置签名
            self._last_todo_sig = None

        # 处理 done
        if done_tools:
            msg = done_tools[0]["params"].get("message", "任务完成")

            # ===== Tool-Done 同轮分离（强制拦截，非提示词约束）=====
            # 现场：LLM 把 file_view/bash/create 与 done 写进同一轮回复。
            # 若此处直接返回 QuitByDone，主循环立即 break，工具结果虽已 append
            # 到 api_messages，LLM 却永远没有下一轮去读它——用户视角就是
            # "读到一半/搜到一半突然停住"。提示词已多次强调仍会复发，故在
            # 引擎层强制丢弃本轮的 done（与上方 has_ask_user 同一处置模式）。
            # 丢弃后不额外注入提醒：工具结果本身就会驱动下一轮继续，
            # 注入只会白烧 token 并打断 LLM 的自然推进。
            if any_effective_tool:
                self._print_info(
                    "[协议纠正] 本轮同时包含工具与 done，已忽略 done，"
                    "先返回工具结果继续推进（done 将在工具结果消化后再判定）"
                )
                self.session.log_dict_info({
                    "role": "system",
                    "content": (
                        f"[协议纠正] 同一轮回复同时包含工具调用与 done，已强制忽略 done"
                        f"（done 内容: {msg[:80]}），待工具结果回灌后由 LLM 重新决策"
                    )
                })
                return ChatOrNot.Continue, tool_exec_info

            self._print_info(msg)
            self.session.log_dict_info({"role": "assistant", "content": msg})

            self._todo_manager.reset()  # 任务完成，清空 todo 列表
            return ChatOrNot.QuitByDone, tool_exec_info

        # self._print_info("no tools")
        return ChatOrNot.Continue, tool_exec_info


    def _save_turn_memory(self, turn: int, user_input: str,
                          reasoning_content: str, remaining_text: str,
                          tool_exec_info: list) -> None:
        """将整轮对话打包为一条完整记忆（用户输入 + LLM 思考 + 应答 + 工具执行）。

        解决原有分开存储导致召回不完整的问题。
        """
        try:
            parts = []  # noqa E352

            # 1. 用户输入
            parts.append(f"[用户输入] {user_input}")

            # 2. LLM 思考过程
            if reasoning_content:
                parts.append(f"[LLM 思考] {reasoning_content}")

            # 3. LLM 应答
            if remaining_text:
                parts.append(f"[LLM 应答] {remaining_text}")

            # 4. 工具执行
            if tool_exec_info:
                for i, info in enumerate(tool_exec_info):
                    parts.append(f"[工具执行{i+1}] 工具: {info['tool']}")
                    parts.append(f"[工具参数{i+1}] {info['params']}")
                    parts.append(f"[工具结果{i+1}] {info['result']}")

            content = "\n\n".join(parts)
            self._memory.add("", content, metadata={
                "turn": turn,
                "has_tools": bool(tool_exec_info),
                "has_reasoning": bool(reasoning_content),
                "query_id": self._query_counter,
                "user_input": user_input,
                "session_id": getattr(self.session, 'session_id', ''),
            })
            self._turn_memory_saved = True  # 优雅退出：本 Query 记忆已保存（中断补存判据）
            logger.debug(f"Turn {turn} 完整记忆已存储，长度: {len(content)}")
        except Exception as e:
            logger.warning(f"完整轮次记忆存储失败: {e}")


    @staticmethod
    def _count_recalled(mem_context: str) -> int:
        """从记忆注入上下文的标记中提取召回的记忆条目数。

        依赖注入文本末尾的 ``[召回数: N]`` 标记，避免逆向解析内容行导致漏匹配。
        """
        if not mem_context:
            return 0
        match = re.search(r"\[召回数:\s*(\d+)\]", mem_context)
        if match:
            return int(match.group(1))
        return 0


    @staticmethod
    def _compress_assistant_message(raw: str) -> str:
        """
        将 assistant 消息中的工具调用压缩为简短形式。
        - <create>...</create>  -> <create path="..." summary="..."/>
        - <str_replace>...</str_replace> -> <str_replace path="..." summary="..."/>
        其他标签原样保留（已足够短）。
        """
        #  import re

        # 压缩 <create> 标签
        def replace_create(match):
            path = match.group(1)
            summary = match.group(2) or ""
            # 可选：截断摘要（但 LLM 已被要求 ≤50 字符，这里暂不处理）
            return f'<create path="{path}" summary="{summary}"/>'

        create_pattern = re.compile(
            r'<create\s+path="([^"]*)"(?:\s+summary="([^"]*)")?\s*>(.*?)</create>',
            re.DOTALL
        )
        compressed = create_pattern.sub(replace_create, raw)

        # 压缩 <str_replace> 标签
        def replace_str_replace(match):
            path = match.group(1)
            summary = match.group(2) or ""
            return f'<str_replace path="{path}" summary="{summary}"/>'

        str_replace_pattern = re.compile(
            r'<str_replace\s+path="([^"]*)"(?:\s+summary="([^"]*)")?\s*>.*?</str_replace>',
            re.DOTALL
        )
        compressed = str_replace_pattern.sub(replace_str_replace, compressed)

        # file_view, bash，原样已足够短，无需改动
        '''
        bash_pattern = re.compile(r'<bash>(.*?)</bash>', re.DOTALL)
        compressed = bash_pattern.sub(r'<bash command="\1"/>', compressed)
        '''

        return compressed
