"""互联网搜索与网页抓取工具（web_search / web_fetch）。

配置驱动的 tools 协议实现（spec/web_search_refactor_spec.md）：
- 端点与密钥来自 model_key.yaml 的 web_search 段（provider 引用命名模型块 +
  search_api / fetch_api 端点声明字符串），代码不硬编码任何厂商端点；
- 参数来自 config.yaml 的 web_search 段（开关/触发模式/隐私词/条数/超时/
  重试/缓存/限流/预算）；
- 富字段搜索结果（rerank_score / snippet / summary / content / site_name /
  site_icon / date / authority），CLI 与 LLM 双通道内容分流；
- web_fetch 单 URL 抓取返回 Markdown，与 file_view（本地文件）互补；
- 双通道输出契约：工具入口返回 {"llm_text": str, "display": dict}，
  llm_text 注入 LLM 上下文（token 预算内），display 供 CLI Rich 面板渲染。

工具协议（LLM 侧，自闭合标签）：

    <web_fetch url="https://example.com/docs"/>

设计约束（遵守项目架构契约）：
- 全同步 requests 调用，禁止 async/await
- api_key 绝不进入 LLM 上下文（仅出现在 HTTP 请求头中）
- 任何异常不中断 Query Loop（工具入口全包裹 try/except）
"""

import hashlib
import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from wdclaude.utility.config_loader import global_cfg

logger = logging.getLogger(__name__)


# ======================== 数据模型（spec 5.2） ========================

@dataclass
class SearchRequest:
    """搜索请求（厂商中立）。"""
    query: str                          # 1~128 Unicode 字符
    count: int = 5                      # 钳制 1~50
    ori_question: str | None = None     # 触发搜索的用户原始问题，<=128 字符
    need_content: bool = False


@dataclass
class SearchResult:
    """单条搜索结果（富字段，厂商中立）。"""
    title: str
    url: str
    snippet: str = ""               # 100~200 字符，面向 CLI 展现
    summary: str = ""               # 600~800 字符，面向模型阅读
    content: str = ""               # need_content=true 时有值
    site_name: str | None = None
    site_icon: str | None = None    # 本期仅留存字段，不渲染图标
    date: str | None = None         # UTC RFC 3339
    authority: int | None = None    # 0~4，-1 未知，>=3 为极权威
    rerank_score: float | None = None   # 0~3，>=1.5 为优质结果


@dataclass
class FetchResult:
    """单 URL 抓取结果。"""
    url: str
    title: str = ""             # 页面标题（API title，缺失为空字符串）
    markdown: str = ""          # Markdown 正文（API content 字段）
    fetched_at: str = ""        # 本地记录的 ISO 时间
    truncated: bool = False     # 注入预算截断标记


# ======================== 配置模型与加载（spec 4 章） ========================

@dataclass
class WebSearchConfig:
    """web_search / web_fetch 全量配置。

    敏感字段（provider / search_api / fetch_api）来自 model_key.yaml 的
    web_search 段；参数字段来自 config.yaml 的 web_search 段（经 config_loader
    深度合并后统一从 global_cfg.web_search 读取）。
    """
    # ----- model_key.yaml web_search 段（端点与密钥引用） -----
    provider: str = ""           # 命名模型块引用（如 "DeepSeek-main"）
    search_api: str = ""         # 搜索端点声明 "METHOD URL"
    fetch_api: str = ""          # 抓取端点声明（选填，缺失仅禁用 web_fetch）
    # ----- provider 引用块解析结果 -----
    api_key: str = ""
    base_url: str = ""
    # ----- config.yaml web_search 参数段 -----
    enabled: bool = False
    trigger_mode: str = "private"        # auto / explicit / private
    sensitive_words: tuple = ()          # 受保护词汇（query / URL 命中即拒绝）
    max_results: int = 5                 # 默认返回条数（映射 API count，钳制 1~50）
    need_content: bool = False           # 是否随搜索返回网页正文 content
    timeout: int = 10                    # HTTP 超时（秒），search 与 fetch 共用
    max_retries: int = 2                 # 失败重试次数（指数退避）
    cache_ttl: int = 3600                # 缓存有效期（秒），0 禁用；search/fetch 独立缓存
    token_budget: int = 3000             # 单次搜索结果注入 token 预算
    fetch_token_budget: int = 3000       # 单次抓取内容注入 token 预算
    min_interval: float = 2.0            # 两次真实请求（search/fetch 合计）最小间隔（秒）
    max_calls_per_session: int = 10      # 会话级调用上限（search/fetch 分别计数），0 = 不限
    fetch_fail_ttl: int = 600            # 抓取失败负面缓存有效期（秒），0 禁用
    domain_fail_threshold: int = 3       # 同域名连续抓取失败达到该次数即熔断，0 禁用
    domain_fail_cooldown: int = 600      # 域名熔断冷却时长（秒），到期自动解封


def _to_bool(value) -> bool:
    """兼容 YAML 中的布尔值与字符串写法（true/"True"/"1"/"on"）。"""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "on")
    return bool(value)


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int_safe(value):
    """宽容 int 转换：失败返回 None（用于响应字段归一化）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_safe(value):
    """宽容 float 转换：失败返回 None（用于响应字段归一化）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ws_config() -> WebSearchConfig:
    """从 global_cfg 读取 web_search 配置段（model_key 与 config 合并后的结果）。

    配置缺失时返回全默认值（不影响启动）；旧形态检测（缺 search_api）
    由工具入口完成并返回配置指引。
    """
    cfg = WebSearchConfig()
    ws = getattr(global_cfg, "web_search", None)
    if ws is None:
        return cfg
    # model_key.yaml 侧字段（新形态：provider 引用 + 端点声明）
    cfg.provider = str(getattr(ws, "provider", "") or "").strip()
    cfg.search_api = str(getattr(ws, "search_api", "") or "").strip()
    cfg.fetch_api = str(getattr(ws, "fetch_api", "") or "").strip()
    # config.yaml 侧参数
    cfg.enabled = _to_bool(getattr(ws, "enabled", cfg.enabled))
    trigger_mode = str(getattr(ws, "trigger_mode", cfg.trigger_mode)).strip().lower()
    cfg.trigger_mode = trigger_mode if trigger_mode in ("auto", "explicit", "private") else "private"
    raw_words = getattr(ws, "sensitive_words", None) or []
    if isinstance(raw_words, str):
        raw_words = [raw_words]
    cfg.sensitive_words = tuple(str(w).strip() for w in raw_words if str(w).strip())
    cfg.max_results = _to_int(getattr(ws, "max_results", cfg.max_results), cfg.max_results)
    cfg.need_content = _to_bool(getattr(ws, "need_content", cfg.need_content))
    cfg.timeout = _to_int(getattr(ws, "timeout", cfg.timeout), cfg.timeout)
    cfg.max_retries = _to_int(getattr(ws, "max_retries", cfg.max_retries), cfg.max_retries)
    cfg.cache_ttl = _to_int(getattr(ws, "cache_ttl", cfg.cache_ttl), cfg.cache_ttl)
    cfg.token_budget = _to_int(getattr(ws, "token_budget", cfg.token_budget), cfg.token_budget)
    cfg.fetch_token_budget = _to_int(
        getattr(ws, "fetch_token_budget", cfg.fetch_token_budget), cfg.fetch_token_budget)
    cfg.min_interval = _to_float(getattr(ws, "min_interval", cfg.min_interval), cfg.min_interval)
    cfg.max_calls_per_session = _to_int(
        getattr(ws, "max_calls_per_session", cfg.max_calls_per_session), cfg.max_calls_per_session)
    cfg.fetch_fail_ttl = _to_int(
        getattr(ws, "fetch_fail_ttl", cfg.fetch_fail_ttl), cfg.fetch_fail_ttl)
    cfg.domain_fail_threshold = _to_int(
        getattr(ws, "domain_fail_threshold", cfg.domain_fail_threshold), cfg.domain_fail_threshold)
    cfg.domain_fail_cooldown = _to_int(
        getattr(ws, "domain_fail_cooldown", cfg.domain_fail_cooldown), cfg.domain_fail_cooldown)
    return cfg


# ======================== 端点声明解析（spec 4.3） ========================

_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}


def parse_endpoint(decl: str, base_url: str = "") -> tuple:
    """解析端点声明字符串 "METHOD URL"。

    规则（spec 4.3）：
    - METHOD：缺省默认 POST；声明其他方法（GET 等）时报错"暂不支持"；
    - URL：必须含协议（http/https），缺协议自动补 https://；
      以 / 开头的相对路径拼接 provider 引用块的 base_url。

    Returns:
        (method, url)；解析失败抛出 ValueError。
    """
    text = (decl or "").strip()
    if not text:
        raise ValueError("端点声明为空")
    tokens = text.split()
    if tokens[0].upper() in _HTTP_METHODS and len(tokens) > 1:
        method = tokens[0].upper()
        url = text[len(tokens[0]):].strip()
    else:
        method = "POST"
        url = text
    if method != "POST":
        raise ValueError(f"端点方法 {method} 暂不支持（当前仅支持 POST）")
    if not url:
        raise ValueError("端点声明缺少 URL")
    if url.startswith("/"):
        if not base_url:
            raise ValueError("相对路径端点需要 provider 引用块的 base_url 作为拼接基准")
        url = base_url.rstrip("/") + url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return method, url


def _resolve_provider_ref(cfg: WebSearchConfig) -> tuple:
    """解析 provider 引用块（spec 4.2 解析规则 1）。

    引用块必须存在且含非空 api_key（模板占位视为空），成功时把 api_key 与
    base_url 写回 cfg（base_url 作为相对路径端点的拼接基准）。

    Returns:
        (ok, err_msg)：ok=False 时 err_msg 为给 LLM 的错误说明。
    """
    if not cfg.provider:
        return False, "provider 未配置（model_key.yaml web_search.provider）"
    block = getattr(global_cfg, cfg.provider, None)
    if block is None:
        return False, f"provider 引用块 '{cfg.provider}' 未在 model_key.yaml 顶层定义"
    api_key = str(getattr(block, "api_key", "") or "").strip()
    if (not api_key or "在此填写" in api_key or api_key.startswith("sk-xxx")
            or not api_key.strip("*")):
        return False, f"provider 引用块 '{cfg.provider}' 的 api_key 为空或仍为模板占位"
    cfg.api_key = api_key
    cfg.base_url = str(getattr(block, "base_url", "") or "").strip()
    return True, ""


# ======================== 隐私过滤（代码级硬约束，spec 9.3 L2） ========================

# 高置信敏感模式：命中即拒绝，所有 trigger_mode 下均生效。
# 路径/密钥类内容对搜索本身无价值，出现在 query 里只会泄露项目内部信息。
_SENSITIVE_PATTERNS = (
    (re.compile(r'[A-Za-z]:[\\/]'), "盘符路径（如 D:\\ 或 D:/）"),
    (re.compile(r'\w+\\\w+'), "反斜杠路径片段"),
    (re.compile(r'\b(?:sk|tvly|gsk)-[A-Za-z0-9_\-]{12,}'), "API Key 形态字符串（sk-/tvly-/gsk- 前缀）"),
    (re.compile(r'\b[0-9a-f]{32}\.[A-Za-z0-9_\-]{8,}\b'), "GLM Key 形态字符串（32位hex.后缀）"),
    (re.compile(r'\bBearer\s+\S+', re.IGNORECASE), "认证头 Bearer"),
)


def check_query_privacy(query: str, sensitive_words: tuple = ()) -> str:
    """检查 query 是否含敏感内容。返回命中原因，干净则返回空字符串。

    规则（所有模式生效）：
    1. 路径/密钥形态：盘符路径、反斜杠路径、sk-/tvly- 类 Key、GLM Key、Bearer 头
    2. 用户自定义 sensitive_words：不区分大小写的子串匹配
    """
    for pat, desc in _SENSITIVE_PATTERNS:
        if pat.search(query):
            return desc
    for w in sensitive_words:
        if w and w.lower() in query.lower():
            return f"受保护词汇 '{w}'"
    return ""


# ======================== fetch URL 安全校验（spec 7.3） ========================

_PRIVATE_HOSTNAMES = {"localhost"}


def _check_fetch_url(url: str, sensitive_words: tuple = ()) -> str:
    """fetch URL 安全校验。返回命中原因，合法返回空字符串。

    拒绝清单（代码级硬约束）：
    - 非 http/https 协议（file://、ftp:// 等）；
    - 本地与内网地址：localhost、127.0.0.0/8、10.0.0.0/8、172.16.0.0/12、
      192.168.0.0/16、169.254.0.0/16、::1、*.local（字面 IP 判定，
      不做 DNS 解析——解析结果可被 DNS rebinding 绕过且增加同步延迟）；
    - 命中 sensitive_words 的 URL；
    - API 侧 url 约束的本地预检：长度超 8192、缺 host、携带用户名/密码。
    """
    text = (url or "").strip()
    if not text:
        return "URL 为空"
    if len(text) > 8192:
        return "URL 长度超过 8192 字符"
    try:
        parsed = urlparse(text)
    except ValueError as e:
        return f"URL 无法解析: {e}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return f"非 http/https 协议（{scheme or '协议缺失'}）"
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return "URL 缺少 host"
    if parsed.username or parsed.password:
        return "URL 携带用户名/密码（禁止 user:pass@host 形态）"
    if host in _PRIVATE_HOSTNAMES or host.endswith(".localhost"):
        return f"本地地址（{host}）禁止抓取"
    if host.endswith(".local"):
        return f"本地域名（*.local）禁止抓取"
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None and not addr.is_global:
        return f"内网/本地 IP（{addr}）禁止抓取"
    for w in sensitive_words:
        if w and w.lower() in text.lower():
            return f"受保护词汇 '{w}'"
    return ""


# ======================== 基础设施：缓存 / 限流 / 会话计数（spec 5.5） ========================

class NamespacedKVCache:
    """本地 JSON 文件 KV 缓存（TTL 控制 + 容量淘汰）。

    search 与 fetch 各持有一个实例（不同文件路径），条目按全字段序列化。
    """

    def __init__(self, cache_path: Path, ttl: int, max_entries: int = 200):
        self.path = Path(cache_path)
        self.ttl = ttl
        self.max_entries = max_entries
        self._data = {}
        self._load()


    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, ValueError) as e:
                logger.warning(f"缓存读取失败，忽略: {e}")
                self._data = {}


    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"缓存写入失败: {e}")


    def get(self, key: str):
        """命中返回条目 value，未命中/过期返回 None。"""
        if self.ttl <= 0:
            return None
        entry = self._data.get(key)
        if not isinstance(entry, dict):
            return None
        if time.time() - entry.get("ts", 0) > self.ttl:
            self._data.pop(key, None)
            return None
        return entry.get("value")


    def put(self, key: str, value):
        if self.ttl <= 0:
            return
        self._data[key] = {"ts": time.time(), "value": value}
        # 容量控制：超限时淘汰最旧的一半条目
        if len(self._data) > self.max_entries:
            sorted_items = sorted(self._data.items(), key=lambda kv: kv[1].get("ts", 0))
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                self._data.pop(k, None)
        self._save()


class RateLimiter:
    """最小间隔限流：search 与 fetch 共用同一实例，防止 LLM 连续刷请求耗尽配额。"""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call = 0.0


    def wait(self):
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


# 会话级调用计数器：模块级单例（spec 5.5），search 与 fetch 分别计数；
# 进程内跨 Query 累积，新 Session 由 query_loop.new_session() 调用重置
_session_call_counts = {"web_search": 0, "web_fetch": 0}


def reset_session_counters():
    """重置会话级调用计数、抓取失败缓存与域名熔断（新 Session 时由 query_loop 调用）。

    失败缓存与熔断虽是进程级优化，但语义上属于「当前会话的已知坏地址」，
    新会话应当重新给这些站点一次机会（站点可能已恢复，或换了网络环境）。
    """
    for k in _session_call_counts:
        _session_call_counts[k] = 0
    _fetch_fail_cache.clear()
    global _breaker_instance
    _breaker_instance = None


def _check_session_limit(tool: str, cfg: WebSearchConfig) -> tuple:
    """会话级调用上限检查（spec 9.3 L2 硬守卫）。

    Returns:
        (hit, notice)：hit=True 时 notice 为给 LLM 的提示（基于已有信息作答）。
    """
    limit = cfg.max_calls_per_session
    if limit <= 0:
        return False, ""
    used = _session_call_counts.get(tool, 0)
    if used >= limit:
        return True, (
            f"[BLOCKED] {tool} 已达会话级调用上限（{limit} 次）。"
            f"请基于已有信息作答，并向用户说明联网查询配额已用尽。"
        )
    return False, ""


# 模块级单例（缓存 / 限流），进程生命周期
_search_cache_instance = None
_fetch_cache_instance = None
_rate_limiter_instance = None


def _get_search_cache(cfg: WebSearchConfig) -> NamespacedKVCache:
    global _search_cache_instance
    if _search_cache_instance is None:
        cache_path = Path(global_cfg.base_path.logs_root) / "web_search_cache.json"
        _search_cache_instance = NamespacedKVCache(cache_path, cfg.cache_ttl)
    return _search_cache_instance


def _get_fetch_cache(cfg: WebSearchConfig) -> NamespacedKVCache:
    global _fetch_cache_instance
    if _fetch_cache_instance is None:
        cache_path = Path(global_cfg.base_path.logs_root) / "web_fetch_cache.json"
        _fetch_cache_instance = NamespacedKVCache(cache_path, cfg.cache_ttl)
    return _fetch_cache_instance


def _get_rate_limiter(cfg: WebSearchConfig) -> RateLimiter:
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter(cfg.min_interval)
    return _rate_limiter_instance


class DomainCircuitBreaker:
    """域名级熔断：同一站点连续抓取失败 N 次后，冷却期内直接拒绝。

    动机：一次问答里 LLM 常对同一站点连续尝试多个 URL（维基中文/英文/
    移动版、同一站的不同路径）。每次失败都要走满重试 + 指数退避，
    单轮耗时可达数十秒，而结果必然相同。熔断把这类重复失败压缩为
    一次 O(1) 判定，并把「换源」信号明确传给 LLM。
    """

    def __init__(self, threshold: int, cooldown: int):
        self.threshold = threshold
        self.cooldown = cooldown
        self._fails = {}      # host -> 连续失败次数
        self._open_until = {}  # host -> 熔断解除时间戳


    @staticmethod
    def host_of(url: str) -> str:
        try:
            return (urlparse(url).hostname or "").lower()
        except ValueError:
            return ""


    def check(self, url: str) -> str:
        """熔断中返回剩余秒数的说明文本，正常返回空字符串。"""
        if self.threshold <= 0 or self.cooldown <= 0:
            return ""
        host = self.host_of(url)
        until = self._open_until.get(host)
        if not until:
            return ""
        remain = until - time.time()
        if remain <= 0:
            self._open_until.pop(host, None)
            self._fails.pop(host, None)
            return ""
        return (f"{host} 已连续抓取失败 {self.threshold} 次，域名熔断中"
                f"（约 {int(remain)} 秒后自动解除）")


    def record_fail(self, url: str):
        if self.threshold <= 0:
            return
        host = self.host_of(url)
        if not host:
            return
        cnt = self._fails.get(host, 0) + 1
        self._fails[host] = cnt
        if cnt >= self.threshold:
            self._open_until[host] = time.time() + self.cooldown
            logger.warning(f"域名熔断开启: {host}（连续失败 {cnt} 次）")


    def record_ok(self, url: str):
        host = self.host_of(url)
        if host:
            self._fails.pop(host, None)
            self._open_until.pop(host, None)


_breaker_instance = None


def _get_breaker(cfg: WebSearchConfig) -> DomainCircuitBreaker:
    global _breaker_instance
    if _breaker_instance is None:
        _breaker_instance = DomainCircuitBreaker(
            cfg.domain_fail_threshold, cfg.domain_fail_cooldown)
    return _breaker_instance


# 抓取失败负面缓存：进程内短期记忆（不落盘、不进 KV 缓存文件）。
# 与 KV 缓存分开实现，因为失败是瞬态信息，不值得持久化污染缓存文件。
_fetch_fail_cache = {}


def _fetch_fail_get(url: str, ttl: int):
    """命中返回失败说明文本，未命中/过期返回 None。"""
    if ttl <= 0:
        return None
    entry = _fetch_fail_cache.get(url)
    if not entry:
        return None
    ts, msg = entry
    if time.time() - ts > ttl:
        _fetch_fail_cache.pop(url, None)
        return None
    return msg


def _fetch_fail_put(url: str, msg: str, ttl: int):
    if ttl <= 0:
        return
    _fetch_fail_cache[url] = (time.time(), msg)
    if len(_fetch_fail_cache) > 200:
        oldest = sorted(_fetch_fail_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in oldest[:100]:
            _fetch_fail_cache.pop(k, None)


def _search_cache_key(query: str, count: int, need_content: bool) -> str:
    """缓存键：query + count + need_content 三维度（spec 5.5）。"""
    return hashlib.md5(f"{query}|{count}|{need_content}".encode("utf-8")).hexdigest()


def _fetch_cache_key(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


# ======================== 服务层：抽象 + tools 协议实现（spec 5.3） ========================

class WebApiError(Exception):
    """tools 协议 API 错误（携带 HTTP 状态码与归一化消息）。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class WebSearchService:
    """搜索/抓取服务门面抽象：search 与 fetch 两个能力原语，数据模型厂商中立。"""

    def search(self, req: SearchRequest) -> list:
        raise NotImplementedError


    def fetch(self, url: str) -> FetchResult:
        raise NotImplementedError


class ToolsApiSearchService(WebSearchService):
    """配置驱动的 tools 协议实现（本期唯一实现）。

    端点与密钥全部来自配置；请求体/响应体按 tools 协议构造与归一化。
    未来他厂若暴露同构 API（POST JSON + Bearer + results 数组），零代码改动
    即可接入；若协议异构，则新增 WebSearchService 实现类并经配置选择。
    """

    # 可重试状态码（spec 12 章：408/429/500/504 等待后重试；其余 4xx 不重试）
    _RETRYABLE_STATUS = {408, 429, 500, 504}

    def __init__(self, cfg: WebSearchConfig):
        self.cfg = cfg
        self._search_method, self._search_url = parse_endpoint(cfg.search_api, cfg.base_url)
        self._fetch_method = ""
        self._fetch_url = ""
        if cfg.fetch_api:
            self._fetch_method, self._fetch_url = parse_endpoint(cfg.fetch_api, cfg.base_url)


    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }


    @staticmethod
    def _extract_error_message(resp) -> str:
        """从 OpenAI 兼容错误体提取 error.message（spec 8.png 错误格式）。

        错误体形如 {"error": {"message", "type", "param": null, "code"}}；
        HTTPError 时优先提取 error.message 呈现给 LLM。
        """
        try:
            data = resp.json()
        except ValueError:
            return (resp.text or "")[:300] or "（无响应体）"
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or "").strip()
                if msg:
                    detail = []
                    if err.get("type"):
                        detail.append(f"type={err['type']}")
                    if err.get("code") is not None:
                        detail.append(f"code={err['code']}")
                    suffix = f"（{', '.join(detail)}）" if detail else ""
                    return f"{msg}{suffix}"
            if data.get("message"):
                return str(data["message"])[:300]
        return str(data)[:300]


    def _post_json(self, url: str, payload: dict) -> dict:
        """带指数退避重试的同步 POST，返回解析后的 JSON dict。

        - 408/429/500/504/超时/连接错误 → 指数退避重试（1s, 2s），上限 max_retries；
        - 400/401/402/403/404/413 等 → 不重试，抛 WebApiError；
        - 200 但响应非 JSON → 抛 WebApiError（附原始响应前 500 字符）。
        """
        last_err = None
        attempts = 0
        for attempt in range(self.cfg.max_retries + 1):
            attempts = attempt + 1
            try:
                resp = requests.post(url, json=payload, headers=self._headers(),
                                     timeout=self.cfg.timeout)
            except requests.exceptions.Timeout:
                last_err = (
                    f"请求超时（单次上限 {self.cfg.timeout}s，第 {attempts} 次尝试）。"
                    f"该 URL 指向的站点响应过慢或抓取端点不可达；"
                    f"可提高 web_search.timeout 后重试，或改抓该故事的镜像源/缓存页"
                )
            except requests.exceptions.SSLError as e:
                last_err = f"TLS 握手失败（证书或代理拦截）: {e}"
            except requests.exceptions.ConnectionError as e:
                last_err = f"连接失败（DNS 解析/网络连通性/代理设置）: {e}"
            except requests.exceptions.RequestException as e:
                last_err = f"请求异常: {e}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise WebApiError(
                            200,
                            f"响应解析失败（非 JSON）: {e}；"
                            f"原始响应前 500 字符: {(resp.text or '')[:500]}"
                        ) from e
                msg = self._extract_error_message(resp)
                if resp.status_code in self._RETRYABLE_STATUS:
                    last_err = f"HTTP {resp.status_code}: {msg}"
                else:
                    raise WebApiError(resp.status_code, msg)
            if attempt < self.cfg.max_retries:
                time.sleep(2 ** attempt)  # 指数退避：1s, 2s
        raise RuntimeError(f"{last_err or '未知请求错误'}（实际尝试 {attempts} 次）")


    # ---- 协议原语 ----

    def _do_search(self, req: SearchRequest) -> dict:
        """构造搜索请求体并 POST search_api 端点（spec 6.1 请求映射）。

        query 原样透传（LLM 可使用搜索语法）；ori_question 仅在非 private
        模式且非空时发送；localization 本期预留不发送。
        """
        payload = {
            "query": req.query,
            "count": req.count,
            "need_content": req.need_content,
        }
        if req.ori_question:
            payload["ori_question"] = req.ori_question
        return self._post_json(self._search_url, payload)


    def _do_fetch(self, url: str) -> dict:
        """构造抓取请求体并 POST fetch_api 端点（spec 7.2）。"""
        if not self._fetch_url:
            raise WebApiError(0, "fetch 端点未配置（model_key.yaml web_search.fetch_api）")
        return self._post_json(self._fetch_url, {"url": url})


    # ---- 门面接口 ----

    def search(self, req: SearchRequest) -> list:
        data = self._do_search(req)
        # 业务层错误：部分服务以 200 + error 字段返回
        if isinstance(data, dict) and data.get("error"):
            raise WebApiError(200, self._extract_error_message(_FakeJson(data)))
        items = _extract_results_array(data)
        return [_map_result_item(item) for item in items][: req.count]


    def fetch(self, url: str) -> FetchResult:
        data = self._do_fetch(url)
        if isinstance(data, dict) and data.get("error"):
            raise WebApiError(200, self._extract_error_message(_FakeJson(data)))
        title = ""
        markdown = ""
        if isinstance(data, dict):
            title = str(data.get("title") or "").strip()
            # content 为确认字段；缺失时按 markdown / text 候选键兜底（schema 演进防御）
            for key in ("content", "markdown", "text"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    markdown = val
                    break
        if not markdown:
            raise WebApiError(
                200,
                "fetch 响应解析失败：缺少 content/markdown/text 字段；"
                f"原始响应前 500 字符: {str(data)[:500]}"
            )
        return FetchResult(url=url, title=title, markdown=markdown,
                           fetched_at=datetime.now().isoformat(timespec="seconds"))


class _FakeJson:
    """让 _extract_error_message 复用于已解析的 dict（业务层 200 + error 场景）。"""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    @property
    def text(self):
        return str(self._data)


# ======================== 响应归一化（spec 5.4，宽容解析） ========================

def _extract_results_array(data) -> list:
    """results 数组提取：候选路径依序探测，首个命中生效。

    (1) data["results"]（DeepSeek 形态）
    (2) data["data"]["webPages"]["value"]（bocha 形态，兼容预留）
    (3) data["choices"][0]["message"]["web_search"]（zhipu 形态，兼容预留）
    """
    if not isinstance(data, dict):
        return []
    for getter in (
        lambda d: d.get("results"),
        lambda d: ((d.get("data") or {}).get("webPages") or {}).get("value"),
        lambda d: ((d.get("choices") or [{}])[0].get("message") or {}).get("web_search"),
    ):
        try:
            val = getter(data)
        except (AttributeError, IndexError, TypeError):
            continue
        if isinstance(val, list):
            return val
    return []


# 单条结果字段映射的候选键（首个非空命中）
_FIELD_CANDIDATES = {
    "title": ("title", "name"),
    "url": ("url", "link"),
    "snippet": ("snippet", "content"),
    "summary": ("summary", "snippet"),
    "content": ("content",),
    "site_name": ("site_name", "siteName", "source"),
    "site_icon": ("site_icon", "siteIcon", "favicon"),
    "date": ("date", "publish_date", "publishedAt"),
}


def _map_result_item(item) -> SearchResult:
    """单条结果归一化：候选键依序探测（首个非空命中），缺失容忍为空值。"""
    if not isinstance(item, dict):
        return SearchResult(title="", url="")

    def pick(candidates):
        for key in candidates:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    authority = _to_int_safe(item.get("authority"))
    rerank_raw = item.get("rerank_score")
    if rerank_raw is None:
        rerank_raw = item.get("score")
    return SearchResult(
        title=pick(_FIELD_CANDIDATES["title"]) or "",
        url=pick(_FIELD_CANDIDATES["url"]) or "",
        snippet=pick(_FIELD_CANDIDATES["snippet"]) or "",
        summary=pick(_FIELD_CANDIDATES["summary"]) or "",
        content=pick(_FIELD_CANDIDATES["content"]) or "",
        site_name=pick(_FIELD_CANDIDATES["site_name"]),
        site_icon=pick(_FIELD_CANDIDATES["site_icon"]),
        date=pick(_FIELD_CANDIDATES["date"]),
        authority=authority,
        rerank_score=_to_float_safe(rerank_raw),
    )


_RESULT_FIELDS = ("title", "url", "snippet", "summary", "content",
                  "site_name", "site_icon", "date", "authority", "rerank_score")


def _result_to_dict(r: SearchResult) -> dict:
    """SearchResult 全字段序列化（缓存条目，spec 5.5：不再只存三字段）。"""
    return {f: getattr(r, f) for f in _RESULT_FIELDS}


def _result_from_dict(d: dict) -> SearchResult:
    return SearchResult(
        title=str(d.get("title") or ""),
        url=str(d.get("url") or ""),
        snippet=str(d.get("snippet") or ""),
        summary=str(d.get("summary") or ""),
        content=str(d.get("content") or ""),
        site_name=d.get("site_name"),
        site_icon=d.get("site_icon"),
        date=d.get("date"),
        authority=d.get("authority"),
        rerank_score=d.get("rerank_score"),
    )


def _fetch_result_to_dict(fr: FetchResult) -> dict:
    return {"url": fr.url, "title": fr.title, "markdown": fr.markdown,
            "fetched_at": fr.fetched_at}


def _fetch_result_from_dict(d: dict) -> FetchResult:
    return FetchResult(
        url=str(d.get("url") or ""),
        title=str(d.get("title") or ""),
        markdown=str(d.get("markdown") or ""),
        fetched_at=str(d.get("fetched_at") or ""),
    )


# ======================== 结果后处理与格式化 ========================

# token→字符换算系数：中英混排约 1 token ≈ 2.5 字符。
# 纯中文约 1 token ≈ 1.5~2 字符，纯英文约 1 token ≈ 4 字符，取折中值，
# 避免旧实现（固定 *2）在英文长页面上低估可用字符数、在中文页面上高估。
_CHARS_PER_TOKEN = 2.5


def _truncate_middle(text: str, limit: int, head_ratio: float = 0.75) -> tuple:
    """头尾保留截断：返回 (截断后文本, 是否截断, 省略字符数)。

    纯头部截断会把页面尾部的数据表、签表、奖金列表整段丢弃，这些恰恰是
    结构化信息最密集的部分。改为「留头 75% + 留尾 25% + 中间省占位」，
    让模型既看到开篇结论，也能读到收尾的数据表。
    """
    if limit <= 0 or len(text) <= limit:
        return text, False, 0
    marker_reserve = 120
    keep = max(200, limit - marker_reserve)
    head_len = int(keep * head_ratio)
    tail_len = max(0, keep - head_len)
    omitted = len(text) - head_len - tail_len
    if omitted <= 0:
        return text, False, 0
    marker = (
        f"\n\n…（此处省略 {omitted} 字符，因单次注入 token 预算限制；"
        f"首尾已完整保留）…\n\n"
    )
    return text[:head_len] + marker + text[-tail_len:], True, omitted


def _dedup(results: list) -> list:
    """按 URL 去重（忽略末尾斜杠与大小写）。

    空 URL 的结果保留（按标题去重），非空 URL 照常按 URL 去重——
    防止"全部召回但无 url 字段"的结果被整条静默丢弃。
    """
    seen_urls = set()
    seen_titles = set()
    out = []
    for r in results:
        url_key = (r.url or "").rstrip("/").lower()
        if url_key:
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            out.append(r)
        else:
            title_key = (r.title or "").strip().lower()
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            out.append(r)
    return out


def _format_search_llm_text(query: str, results: list, provider_name: str,
                            token_budget: int, cached: bool) -> tuple:
    """将搜索结果格式化为编号列表并按 token 预算截断（spec 11 章）。

    token 估算：中文约 1 token ≈ 2 字符，按字符预算 = token_budget * 2 保守控制。
    摘要优先用 summary（为模型而生的长摘要），缺失回退 snippet。

    Returns:
        (llm_text, 实际注入条数)——注入条数供双通道一致性标注使用。
    """
    if not results:
        return (
            f"搜索完成但未返回任何结果（后端: {provider_name}）。\n查询: {query}\n"
            f"可尝试换更具体的关键词或使用搜索语法（site: / lang: / after: 等）；"
            f"若换词后仍为空，请基于已有知识回答并声明未能联网获取实时数据。"
        ), 0
    char_budget = max(500, int(token_budget * _CHARS_PER_TOKEN))
    entries = []
    used = 0
    for i, r in enumerate(results, 1):
        summary = r.summary or r.snippet or "（无摘要）"
        meta_parts = []
        if r.date:
            meta_parts.append(f"日期: {r.date}")
        if r.site_name:
            meta_parts.append(f"站点: {r.site_name}")
        if r.authority is not None and r.authority >= 0:
            meta_parts.append(f"权威: {r.authority}/4")
        if r.rerank_score is not None:
            meta_parts.append(f"排序分: {r.rerank_score}")
        head = f"{i}. {r.title or '（无标题）'}\n   URL: {r.url}"
        if meta_parts:
            head += "\n   " + " | ".join(meta_parts)
        entry = head + f"\n   摘要: {summary}"
        if r.content:
            entry += f"\n   正文: {r.content}"
        remain = char_budget - used
        if remain <= len(head) + 60:
            break
        if len(entry) > remain:
            # 末条不整条丢弃：保留标题与元信息，裁切摘要正文后再收尾
            keep = max(80, remain - len(head) - 60)
            cut = summary[:keep]
            entries.append(f"{head}\n   摘要: {cut}…（本条摘要因注入预算截断）")
            used = char_budget
            break
        entries.append(entry)
        used += len(entry)

    header = (f'[web_search] 查询 "{query}"（后端 {provider_name}'
              f'{"，来自本地缓存" if cached else ""}，{len(results)} 条）')
    if len(entries) < len(results):
        header += f"，因 token 预算仅注入前 {len(entries)} 条"
    return header + ":\n\n" + "\n\n".join(entries), len(entries)


def _fetch_preview(markdown: str, max_lines: int = 3) -> str:
    """抓取内容的首 N 行预览（每行截断 120 字符，供 CLI 面板展示）。"""
    lines = [ln.strip() for ln in (markdown or "").splitlines() if ln.strip()]
    return "\n".join(ln[:120] for ln in lines[:max_lines])


# ======================== ori_question 附加（spec 9.4） ========================

# 最近一条用户消息（query_loop 每 Query 开始时调用 set_last_user_input 更新）
_last_user_input = ""


def set_last_user_input(text: str):
    """记录最近一条用户消息，供 ori_question 自动附加。"""
    global _last_user_input
    _last_user_input = (text or "").strip()


def _build_ori_question(cfg: WebSearchConfig) -> str:
    """构造 ori_question（spec 9.4 附加策略）。

    - auto / explicit 模式：附加最近一条用户消息（剥离 @ 引用/路径/标签噪声，
      截断 128 字符）；附加前过隐私过滤，命中敏感内容则省略该字段
      （不阻断搜索，因 query 本身已通过过滤）；
    - private 模式：不附加（用户消息可能含项目细节，防止泄露）。
    """
    if cfg.trigger_mode == "private":
        return ""
    raw = _last_user_input
    if not raw:
        return ""
    cleaned = re.sub(r'[A-Za-z]:[/\\][^\s，。,；;）)]*', ' ', raw)   # 盘符路径
    cleaned = re.sub(r'@\S+', ' ', cleaned)                          # @ 引用
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)                       # XML 标签
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return ""
    cleaned = cleaned[:128]
    if check_query_privacy(cleaned, cfg.sensitive_words):
        return ""
    return cleaned


# ======================== 双通道输出构造（spec 10.2 / 11 章） ========================

def _search_display_results(results: list) -> list:
    """display.results：CLI 展示字段（snippet 给人，spec 6.2 双通道分流）。"""
    out = []
    for r in results:
        snippet = r.snippet or (r.summary[:200] if r.summary else "")
        out.append({
            "title": r.title,
            "url": r.url,
            "snippet": snippet,
            "date": r.date,
            "site_name": r.site_name,
            "authority": r.authority,
            "rerank_score": r.rerank_score,
        })
    return out


def _search_output(cfg: WebSearchConfig, query: str, results: list, cached: bool) -> dict:
    """搜索成功路径的双通道输出（llm_text 与 display 信息保持一致）。"""
    llm_text, injected = _format_search_llm_text(
        query, results, cfg.provider, cfg.token_budget, cached)
    notices = []
    if not results:
        notices.append("无结果，建议换更具体的关键词或使用搜索语法（site: / lang: / after:）重试")
    elif injected < len(results):
        notices.append(f"因 token 预算仅注入前 {injected} 条")
    display = {
        "tool": "web_search",
        "query_or_url": query,
        "provider": cfg.provider,
        "cached": cached,
        "results": _search_display_results(results),
        "notice": "；".join(notices),
        "error": False,
    }
    return {"llm_text": llm_text, "display": display}


def _fetch_output(cfg: WebSearchConfig, fr: FetchResult, cached: bool) -> dict:
    """抓取成功路径的双通道输出：markdown 按 fetch_token_budget 首部截断（spec 7.4）。"""
    char_budget = max(500, int(cfg.fetch_token_budget * _CHARS_PER_TOKEN))
    full_chars = len(fr.markdown)
    markdown, truncated, omitted = _truncate_middle(fr.markdown, char_budget)
    title_part = f"{fr.title} | " if fr.title else ""
    if truncated:
        trunc_part = f"，首尾保留注入，中段省略 {omitted} 字符"
        notice = (f"内容 {full_chars} 字符超出单次预算（约 {cfg.fetch_token_budget} token），"
                  f"已保留首尾、省略中段 {omitted} 字符；如需完整内容可分段抓取或提高 "
                  f"web_search.fetch_token_budget")
    else:
        trunc_part = ""
        notice = ""
    llm_text = (f"[web_fetch] {title_part}{fr.url}"
                f"（Markdown {full_chars} 字符{trunc_part}）:\n{markdown}")
    display = {
        "tool": "web_fetch",
        "query_or_url": fr.url,
        "title": fr.title,
        "provider": cfg.provider,
        "cached": cached,
        "char_count": full_chars,
        "truncated": truncated,
        "omitted_chars": omitted,
        "preview": _fetch_preview(markdown),
        "notice": notice,
        "error": False,
    }
    return {"llm_text": llm_text, "display": display}


def _status_display(tool: str, query_or_url: str, notice: str, error: bool = False) -> dict:
    """失败/拦截路径的 display（notice 承载错误信息，spec 10.2 兼容要求）。"""
    d = {
        "tool": tool,
        "query_or_url": query_or_url,
        "provider": "",
        "cached": False,
        "results": [],
        "notice": notice,
        "error": error,
    }
    if tool == "web_fetch":
        d.update({"title": "", "char_count": 0, "truncated": False, "preview": ""})
    return d


def _blocked_output(tool: str, query_or_url: str, notice: str) -> dict:
    return {"llm_text": notice, "display": _status_display(tool, query_or_url, notice, error=True)}


def _error_output(tool: str, query_or_url: str, notice: str) -> dict:
    return {"llm_text": notice, "display": _status_display(tool, query_or_url, notice, error=True)}


def _api_error_text(e: WebApiError, tool: str) -> str:
    """将 API 错误转换为面向 LLM 的指令文本（spec 12 章错误矩阵）。"""
    code = e.status_code
    head = f"[ERROR] {tool} 执行失败: HTTP {code}" if code else f"[ERROR] {tool} 执行失败"
    if code in (400, 413):
        return (f"{head}（请求头或请求体不合法/超限）: {e.message}\n"
                f"请修正参数（count 范围、url 约束等）后再调用，勿原样重试。")
    if code == 401:
        return (f"{head}（认证失败，Key 缺失或无效）: {e.message}\n"
                f"请声明未能联网，勿立即重试；建议用户检查 model_key.yaml 密钥配置。")
    if code == 402:
        return (f"{head}（余额不足）: {e.message}\n"
                f"请声明未能联网，勿立即重试；建议用户先充值后再试。")
    if code == 403:
        if tool == "web_fetch":
            return (f"{head}（fetch 被安全策略拒绝）: {e.message}\n"
                    f"请更换 URL，或改用 web_search 定位其他来源。")
        return f"{head}: {e.message}\n请勿原样重试。"
    if code == 404:
        # 抓取端点的 404 有两类完全不同的成因，混在一起会让 LLM 误判为「网址错了」：
        #  (a) 上游站点确实返回 404（页面不存在/已删除）；
        #  (b) 上游站点反爬拦截、返回非 HTML（PDF/JS 渲染页）或超时，
        #      抓取网关统一以 404 上报「未能获取可用内容」。
        # 后者换 URL 才有意义，前者提高超时或换镜像才有意义，故文案同时给出两路。
        if tool == "web_fetch":
            return (f"{head}（未能获取可用内容：抓取网关未取回正文）: {e.message}\n"
                    f"常见成因两类：a) 源站反爬/需 JS 渲染/返回非 HTML，网关抓不到正文；"
                    f"b) 该 URL 确实已失效。\n"
                    f"处置：先用 web_search 检索同一主题的其他来源再抓取；"
                    f"不要对同一 URL 原样重试；若整站抓取连续失败，改用搜索结果的 summary "
                    f"作答并向用户说明原始页面不可读。")
        return f"{head}: {e.message}\n请勿原样重试。"
    return (f"{head}: {e.message}\n"
            f"若重试仍失败，请基于已有知识回答并声明未能联网。")


# ======================== 工具入口（tool_executor 调用） ========================

def execute_web_search(params: dict) -> dict:
    """web_search 工具执行入口（spec 10.2 双通道契约）。

    Args:
        params: {"query": str, "max_results": int(可选)}

    Returns:
        {"llm_text": str, "display": dict}：llm_text 注入 LLM 上下文
        （token 预算内），display 供 CLI 渲染；失败时同样返回该结构
        （display.notice 承载错误信息，display.error=True）。
    """
    try:
        return _execute_web_search_impl(params)
    except Exception as e:  # noqa BLE001 兜底：任何异常不得中断 Query Loop
        logger.error(f"web_search 未捕获异常: {e}")
        return _error_output(
            "web_search", str((params or {}).get("query", "")),
            f"[ERROR] web_search 异常: {e}\n请基于已有知识回答，并明确告知用户本次未能联网搜索。")


def _execute_web_search_impl(params: dict) -> dict:
    query = str(params.get("query", "") or "").strip()
    if not query:
        return _error_output(
            "web_search", "",
            '[ERROR] web_search 缺少 query 参数。用法: <web_search query="搜索词" max_results="5"/>')

    try:
        max_results = int(params.get("max_results") or 0) or None
    except (TypeError, ValueError):
        max_results = None

    cfg = load_ws_config()

    # 功能开关检查
    if not cfg.enabled:
        return _blocked_output(
            "web_search", query,
            "[BLOCKED] web_search 功能未启用（config.yaml 中 web_search.enabled 为 false）。"
            "请基于已有知识回答，并向用户说明当前无法联网搜索。")

    # 旧形态检测：web_search 段缺 search_api → 功能视为未配置（spec 4.5）
    if not cfg.search_api:
        return _blocked_output(
            "web_search", query,
            "[BLOCKED] web_search 未配置搜索端点：model_key.yaml 的 web_search 段缺 search_api"
            "（旧形态已停止支持）。请按新形态配置：\n"
            "web_search:\n"
            '  provider: "DeepSeek-main"    # 引用本文件中已定义的命名模型块\n'
            '  search_api: "POST https://api.deepseek.com/v1/tools/search"\n'
            '  fetch_api: "POST https://api.deepseek.com/v1/tools/fetch"\n'
            "配置后重启生效。当前请基于已有知识回答，并向用户说明无法联网搜索。")

    # provider 引用块解析（密钥三级查找由 config_loader 完成，此处校验引用有效性）
    ok, err = _resolve_provider_ref(cfg)
    if not ok:
        return _blocked_output(
            "web_search", query,
            f"[BLOCKED] web_search {err}。请检查 model_key.yaml 中 web_search.provider 的引用配置，"
            f"或基于已有知识回答并声明无法联网。")

    # 隐私硬过滤（代码级，优先于缓存与真实请求，所有模式生效）
    leak_reason = check_query_privacy(query, cfg.sensitive_words)
    if leak_reason:
        return _blocked_output(
            "web_search", query,
            f"[BLOCKED] web_search 隐私过滤：query 含{leak_reason}，已拒绝执行。\n"
            f"请将问题改写为面向公网知识的通用描述（如 'Python 3.14 free-threading 稳定性'），"
            f"严禁携带项目路径、文件名、函数名、密钥或内部报错细节，然后重新调用；"
            f"或直接基于已有知识回答。")

    # 参数钳制：query 1~128 字符（超出截断），count 1~50
    query = query[:128]
    count = max_results if max_results is not None else cfg.max_results
    count = max(1, min(count, 50))

    # 缓存命中：直接返回，不消耗配额（键含 need_content 维度）
    cache = _get_search_cache(cfg)
    cache_key = _search_cache_key(query, count, cfg.need_content)
    cached_items = cache.get(cache_key)
    if cached_items:
        results = [_result_from_dict(item) for item in cached_items]
        return _search_output(cfg, query, results, cached=True)

    # 会话级调用上限（L2 硬守卫）
    limit_hit, limit_notice = _check_session_limit("web_search", cfg)
    if limit_hit:
        return _blocked_output("web_search", query, limit_notice)

    # 端点解析（配置声明驱动，spec 4.3）
    try:
        service = ToolsApiSearchService(cfg)
    except ValueError as e:
        return _error_output("web_search", query,
                             f"[ERROR] web_search 端点配置解析失败: {e}")

    req = SearchRequest(query=query, count=count,
                        ori_question=_build_ori_question(cfg) or None,
                        need_content=cfg.need_content)

    # 限流 + 真实请求
    try:
        _get_rate_limiter(cfg).wait()
        _session_call_counts["web_search"] += 1
        results = service.search(req)
    except WebApiError as e:
        return _error_output("web_search", query, _api_error_text(e, "web_search"))
    except RuntimeError as e:
        return _error_output(
            "web_search", query,
            f"[ERROR] web_search 执行失败（已重试 {cfg.max_retries} 次）: {e}\n"
            f"请基于已有知识回答，并明确告知用户本次未能联网搜索，不要立即重试。")
    except Exception as e:  # noqa BLE001 兜底
        logger.error(f"web_search 执行异常: {e}")
        return _error_output(
            "web_search", query,
            f"[ERROR] web_search 异常: {e}\n请基于已有知识回答，并明确告知用户本次未能联网搜索。")

    results = _dedup(results)
    # 空结果不写缓存：换词重试不被 TTL 内的空缓存掩盖（沿用既有策略）
    if results:
        cache.put(cache_key, [_result_to_dict(r) for r in results])
    return _search_output(cfg, query, results, cached=False)


def execute_web_fetch(params: dict) -> dict:
    """web_fetch 工具执行入口（spec 7 章 / 10.2 双通道契约）。

    Args:
        params: {"url": str}

    Returns:
        {"llm_text": str, "display": dict}：markdown 按 fetch_token_budget
        首部截断并明确标注；失败时 display.notice 承载错误信息。
    """
    try:
        return _execute_web_fetch_impl(params)
    except Exception as e:  # noqa BLE001 兜底：任何异常不得中断 Query Loop
        logger.error(f"web_fetch 未捕获异常: {e}")
        return _error_output(
            "web_fetch", str((params or {}).get("url", "")),
            f"[ERROR] web_fetch 异常: {e}\n请换 URL 重试，或声明该页不可读。")


def _execute_web_fetch_impl(params: dict) -> dict:
    url = str(params.get("url", "") or "").strip()
    if not url:
        return _error_output(
            "web_fetch", "",
            '[ERROR] web_fetch 缺少 url 参数。用法: <web_fetch url="https://example.com/docs"/>')

    cfg = load_ws_config()

    if not cfg.enabled:
        return _blocked_output(
            "web_fetch", url,
            "[BLOCKED] web_search 功能未启用（config.yaml 中 web_search.enabled 为 false）。"
            "请基于已有知识回答。")

    # fetch_api 选填：缺失时仅禁用 web_fetch，不影响 web_search（spec 4.2）
    if not cfg.fetch_api:
        return _blocked_output(
            "web_fetch", url,
            "[BLOCKED] web_fetch 未配置抓取端点：model_key.yaml 的 web_search 段缺 fetch_api"
            "（选填项，缺失时仅禁用 web_fetch，不影响 web_search）。")

    ok, err = _resolve_provider_ref(cfg)
    if not ok:
        return _blocked_output(
            "web_fetch", url,
            f"[BLOCKED] web_fetch {err}。请检查 model_key.yaml 中 web_search.provider 的引用配置。")

    # URL 安全校验（spec 7.3，代码级硬约束）
    deny = _check_fetch_url(url, cfg.sensitive_words)
    if deny:
        return _blocked_output(
            "web_fetch", url,
            f"[BLOCKED] web_fetch URL 安全校验未通过：{deny}。\n"
            f"请更换合规的公网 URL，或改用 web_search 定位来源。")

    # 失败负面缓存：同一 URL 短期内不必重复烧配额（命中直接返回旧失败说明）
    fail_hit = _fetch_fail_get(url, cfg.fetch_fail_ttl)
    if fail_hit:
        return _error_output(
            "web_fetch", url,
            f"[ERROR] {fail_hit}\n该 URL 近期已抓取失败，本次直接复用失败结论、未再次请求。"
            f"请更换来源（建议先用 web_search 检索同一主题）或声明该页不可读。")

    # 域名熔断：同站连续失败后，冷却期内不再消耗请求与等待时间
    breaker = _get_breaker(cfg)
    break_msg = breaker.check(url)
    if break_msg:
        return _error_output(
            "web_fetch", url,
            f"[ERROR] web_fetch 跳过：{break_msg}\n"
            f"该域名当前不可抓取，请改用 web_search 结果中的其他站点，"
            f"或直接基于已有搜索结果作答并向用户说明原始页面不可读。")

    # 缓存命中（按 URL 为键，TTL 同 cache_ttl）
    cache = _get_fetch_cache(cfg)
    cache_key = _fetch_cache_key(url)
    cached_entry = cache.get(cache_key)
    if cached_entry:
        fr = _fetch_result_from_dict(cached_entry)
        return _fetch_output(cfg, fr, cached=True)

    # 会话级调用上限
    limit_hit, limit_notice = _check_session_limit("web_fetch", cfg)
    if limit_hit:
        return _blocked_output("web_fetch", url, limit_notice)

    try:
        service = ToolsApiSearchService(cfg)
    except ValueError as e:
        return _error_output("web_fetch", url, f"[ERROR] web_fetch 端点配置解析失败: {e}")

    try:
        _get_rate_limiter(cfg).wait()
        _session_call_counts["web_fetch"] += 1
        fr = service.fetch(url)
    except WebApiError as e:
        text = _api_error_text(e, "web_fetch")
        # 认证/余额/参数类错误属配置问题，不是「这个 URL 不行」，不进熔断计数
        if e.status_code not in (0, 400, 401, 402, 403, 413):
            breaker.record_fail(url)
            _fetch_fail_put(url, f"web_fetch 抓取失败（HTTP {e.status_code}）", cfg.fetch_fail_ttl)
        return _error_output("web_fetch", url, text)
    except RuntimeError as e:
        breaker.record_fail(url)
        _fetch_fail_put(url, f"web_fetch 请求失败（{e}）", cfg.fetch_fail_ttl)
        return _error_output(
            "web_fetch", url,
            f"[ERROR] web_fetch 执行失败（已重试 {cfg.max_retries} 次）: {e}\n"
            f"请换 URL 重试，或基于已有知识回答并声明该页不可读。")
    except Exception as e:  # noqa BLE001 兜底
        logger.error(f"web_fetch 执行异常: {e}")
        breaker.record_fail(url)
        return _error_output(
            "web_fetch", url,
            f"[ERROR] web_fetch 异常: {e}\n请换 URL 重试，或声明该页不可读。")
    breaker.record_ok(url)

    # 全量缓存（截断只发生在注入层，预算调大后缓存仍可提供全文）
    cache.put(cache_key, _fetch_result_to_dict(fr))
    return _fetch_output(cfg, fr, cached=False)
