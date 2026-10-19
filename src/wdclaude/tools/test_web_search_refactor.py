"""web_search / web_fetch 重构测试套件（spec/web_search_refactor_spec.md 第 14 章）。

覆盖要点：
- 配置解析：端点字符串（正常 / 缺方法默认 POST / 缺协议自动补 / 相对路径拼接 /
  非法方法报错）；provider 引用（存在 / 缺失 / api_key 为空 / 模板占位）；
  旧形态检测与禁用提示；
- 请求映射：count 钳制、ori_question 截断与隐私省略、need_content 透传、
  private 模式不发送 ori_question；
- 响应归一化：标准 DeepSeek 形态全字段 / 字段缺失与 null 容忍 / 空数组 /
  候选路径探测（bocha、zhipu 形态样本）；
- Fetch：响应解析（title/content 确认 schema、content 缺失时 markdown / text
  兜底）、URL 约束预检（长度 8192 / 含 host / 禁凭据）、markdown 截断标注、
  7.3 封禁清单逐条用例；
- 基础设施：缓存命中 / 过期 / need_content 维度、fetch 缓存、限流间隔、
  会话计数上限、token 预算截断；
- 双通道：同一次执行产出的 llm_text 与 display 信息一致；
- 隐私：query / ori_question 双通道过滤、sensitive_words；
- HTTP mock 覆盖 12 章错误矩阵（monkeypatch requests.post），含 fetch 特定
  403 / 404 语义与 OpenAI 兼容 error.message 提取；
- CLI 渲染冒烟：display 结构到 Rich 面板无异常。
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wdclaude.tools import web_search as ws


# ======================== 测试夹具 ========================

@pytest.fixture
def cfg_factory(monkeypatch, tmp_path):
    """构造可用的 WebSearchConfig 并隔离全局状态（缓存/限流/计数/用户输入）。

    provider 引用块通过 monkeypatch 挂到 global_cfg 顶层（DeepSeek-main 形态）。
    logs_root 重定向到 tmp_path：缓存文件写进临时目录，防止测试间
    通过真实磁盘缓存串扰（前序测试写入的条目被后续测试命中）。
    """
    # 隔离模块级单例
    monkeypatch.setattr(ws, "_search_cache_instance", None)
    monkeypatch.setattr(ws, "_fetch_cache_instance", None)
    monkeypatch.setattr(ws, "_rate_limiter_instance", None)
    monkeypatch.setattr(ws, "_breaker_instance", None)
    monkeypatch.setattr(ws, "_fetch_fail_cache", {}, raising=False)
    monkeypatch.setattr(ws, "_session_call_counts", {"web_search": 0, "web_fetch": 0})
    monkeypatch.setattr(ws, "_last_user_input", "")
    # 缓存目录隔离：所有测试统一指向临时目录
    monkeypatch.setattr(ws.global_cfg, "base_path",
                        SimpleNamespace(logs_root=str(tmp_path)), raising=False)

    class _Factory:
        def make(self, **overrides):
            cfg = ws.WebSearchConfig(
                provider="DeepSeek-main",
                search_api="POST https://api.deepseek.com/v1/tools/search",
                fetch_api="POST https://api.deepseek.com/v1/tools/fetch",
                api_key="sk-test-key",
                base_url="https://api.deepseek.com",
                enabled=True,
                trigger_mode="private",
                sensitive_words=(),
                max_results=5,
                need_content=False,
                timeout=5,
                max_retries=0,
                cache_ttl=3600,
                token_budget=3000,
                fetch_token_budget=3000,
                min_interval=0.0,
                max_calls_per_session=10,
            )
            for k, v in overrides.items():
                setattr(cfg, k, v)
            return cfg

    return _Factory()


@pytest.fixture
def patched_config(monkeypatch, cfg_factory):
    """把 load_ws_config 替换为返回可定制 cfg 的桩，并挂好 provider 引用块。"""
    holder = {"cfg": cfg_factory.make()}

    def _fake_load():
        return holder["cfg"]

    provider_block = SimpleNamespace(api_key="sk-test-key",
                                     base_url="https://api.deepseek.com",
                                     model_name="deepseek-flash")
    monkeypatch.setattr(ws.global_cfg, "DeepSeek-main", provider_block, raising=False)
    monkeypatch.setattr(ws, "load_ws_config", _fake_load)
    return holder


def _mk_response(status_code=200, json_data=None, text=""):
    """构造 mock requests.Response。"""
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no json")
    resp.text = text
    return resp


def _api_result(title="T", url="https://example.com/a", snippet="s", summary="m",
                rerank_score=2.0, authority=3, date="2026-06-01", site_name="Example",
                site_icon=None, content=""):
    return {"title": title, "url": url, "snippet": snippet, "summary": summary,
            "content": content, "site_name": site_name, "site_icon": site_icon,
            "date": date, "authority": authority, "rerank_score": rerank_score}


# ======================== 端点声明解析（spec 4.3） ========================

class TestParseEndpoint:
    def test_normal(self):
        assert ws.parse_endpoint("POST https://api.deepseek.com/v1/tools/search") == \
            ("POST", "https://api.deepseek.com/v1/tools/search")

    def test_default_post_when_method_missing(self):
        assert ws.parse_endpoint("https://api.deepseek.com/v1/tools/search") == \
            ("POST", "https://api.deepseek.com/v1/tools/search")

    def test_scheme_autofill(self):
        assert ws.parse_endpoint("POST api.deepseek.com/v1/tools/search")[1] == \
            "https://api.deepseek.com/v1/tools/search"

    def test_relative_path_joins_base_url(self):
        method, url = ws.parse_endpoint("POST /v1/tools/search", "https://api.deepseek.com")
        assert method == "POST"
        assert url == "https://api.deepseek.com/v1/tools/search"

    def test_relative_path_without_base_raises(self):
        with pytest.raises(ValueError):
            ws.parse_endpoint("POST /v1/tools/search", "")

    def test_get_not_supported(self):
        with pytest.raises(ValueError, match="暂不支持"):
            ws.parse_endpoint("GET https://api.deepseek.com/v1/tools/search")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ws.parse_endpoint("")

    def test_method_case_insensitive(self):
        assert ws.parse_endpoint("post https://a.com/x") == ("POST", "https://a.com/x")


# ======================== provider 引用解析（spec 4.2） ========================

class TestResolveProviderRef:
    def test_ok(self, monkeypatch, cfg_factory):
        cfg = cfg_factory.make()
        monkeypatch.setattr(
            ws.global_cfg, "DeepSeek-main",
            SimpleNamespace(api_key="sk-real", base_url="https://api.deepseek.com"),
            raising=False)
        ok, err = ws._resolve_provider_ref(cfg)
        assert ok and cfg.api_key == "sk-real"

    def test_missing_block(self, monkeypatch, cfg_factory):
        cfg = cfg_factory.make(provider="NoSuchProvider")
        monkeypatch.delattr(ws.global_cfg, "NoSuchProvider", raising=False)
        ok, err = ws._resolve_provider_ref(cfg)
        assert not ok and "未在 model_key.yaml 顶层定义" in err

    def test_empty_api_key(self, monkeypatch, cfg_factory):
        cfg = cfg_factory.make()
        monkeypatch.setattr(
            ws.global_cfg, "DeepSeek-main",
            SimpleNamespace(api_key="", base_url="https://api.deepseek.com"),
            raising=False)
        ok, err = ws._resolve_provider_ref(cfg)
        assert not ok and "api_key" in err

    @pytest.mark.parametrize("placeholder", [
        "在此填写你的 API Key", "sk-xxxxxx", "***", "****",
    ])
    def test_template_placeholder_rejected(self, monkeypatch, cfg_factory, placeholder):
        cfg = cfg_factory.make()
        monkeypatch.setattr(
            ws.global_cfg, "DeepSeek-main",
            SimpleNamespace(api_key=placeholder, base_url="https://api.deepseek.com"),
            raising=False)
        ok, err = ws._resolve_provider_ref(cfg)
        assert not ok


# ======================== 响应归一化（spec 5.4） ========================

class TestNormalize:
    def test_deepseek_full_fields(self):
        item = _api_result()
        data = {"results": [item]}
        arr = ws._extract_results_array(data)
        assert len(arr) == 1
        r = ws._map_result_item(arr[0])
        assert r.title == "T" and r.url == "https://example.com/a"
        assert r.snippet == "s" and r.summary == "m"
        assert r.site_name == "Example" and r.date == "2026-06-01"
        assert r.authority == 3 and r.rerank_score == 2.0

    def test_null_and_missing_tolerance(self):
        r = ws._map_result_item({"title": None, "url": None, "authority": None,
                                 "rerank_score": None, "site_name": None})
        assert r.title == "" and r.url == ""
        assert r.authority is None and r.rerank_score is None

    def test_candidate_keys(self):
        r = ws._map_result_item({"name": "T2", "link": "https://b.com",
                                 "snippet": None, "summary": None,
                                 "source": "Bocha", "score": "1.5"})
        assert r.title == "T2" and r.url == "https://b.com"
        assert r.site_name == "Bocha" and r.rerank_score == 1.5

    def test_bocha_shape(self):
        data = {"data": {"webPages": {"value": [{"name": "B1", "url": "https://b.com/1"}]}}}
        arr = ws._extract_results_array(data)
        assert len(arr) == 1 and arr[0]["name"] == "B1"

    def test_zhipu_shape(self):
        data = {"choices": [{"message": {"web_search": [
            {"title": "Z1", "link": "https://z.com/1"}]}}]}
        arr = ws._extract_results_array(data)
        assert len(arr) == 1 and arr[0]["link"] == "https://z.com/1"

    def test_empty_results(self):
        assert ws._extract_results_array({"results": []}) == []
        assert ws._extract_results_array({}) == []
        assert ws._extract_results_array(None) == []


# ======================== 工具入口：配置与守卫（spec 12） ========================

class TestSearchGuards:
    def test_missing_query(self):
        out = ws.execute_web_search({})
        assert out["llm_text"].startswith("[ERROR]")
        assert out["display"]["error"] is True

    def test_disabled(self, patched_config):
        patched_config["cfg"].enabled = False
        out = ws.execute_web_search({"query": "python"})
        assert out["llm_text"].startswith("[BLOCKED]")
        assert "未启用" in out["llm_text"]

    def test_old_shape_no_search_api(self, patched_config):
        patched_config["cfg"].search_api = ""
        out = ws.execute_web_search({"query": "python"})
        assert out["llm_text"].startswith("[BLOCKED]")
        assert "新形态" in out["llm_text"]
        assert "search_api" in out["llm_text"]

    def test_privacy_blocked(self, patched_config):
        out = ws.execute_web_search({"query": "D:\\project\\src 崩溃"})
        assert out["llm_text"].startswith("[BLOCKED]")
        assert "隐私过滤" in out["llm_text"]
        assert out["display"]["error"] is True

    def test_sensitive_words_blocked(self, patched_config):
        patched_config["cfg"].sensitive_words = ("InternalProject",)
        out = ws.execute_web_search({"query": "InternalProject 架构"})
        assert "[BLOCKED]" in out["llm_text"]
        assert "受保护词汇" in out["llm_text"]


# ======================== 请求映射（spec 6.1） ========================

class TestRequestMapping:
    def _capture_payload(self, monkeypatch, patched_config, results_payload=None):
        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _mk_response(200, results_payload or {"results": [_api_result()]})

        monkeypatch.setattr(ws.requests, "post", _fake_post)
        return captured

    def test_count_clamped_1_to_50(self, monkeypatch, patched_config):
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "q", "max_results": 500})
        assert captured["payload"]["count"] == 50

    def test_count_min_clamp(self, monkeypatch, patched_config):
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "q", "max_results": 0 or 1})
        assert captured["payload"]["count"] >= 1

    def test_need_content_passthrough(self, monkeypatch, patched_config):
        patched_config["cfg"].need_content = True
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "q"})
        assert captured["payload"]["need_content"] is True

    def test_private_mode_no_ori_question(self, monkeypatch, patched_config):
        patched_config["cfg"].trigger_mode = "private"
        ws.set_last_user_input("DeepSeek V4 发布了吗")
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "DeepSeek V4"})
        assert "ori_question" not in captured["payload"]

    def test_auto_mode_ori_question_attached(self, monkeypatch, patched_config):
        patched_config["cfg"].trigger_mode = "auto"
        ws.set_last_user_input("DeepSeek V4 发布了吗")
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "DeepSeek V4"})
        assert captured["payload"].get("ori_question") == "DeepSeek V4 发布了吗"

    def test_ori_question_truncated_128(self, monkeypatch, patched_config):
        patched_config["cfg"].trigger_mode = "auto"
        ws.set_last_user_input("x" * 300)
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "q"})
        assert len(captured["payload"]["ori_question"]) <= 128

    def test_ori_question_privacy_omitted(self, monkeypatch, patched_config):
        patched_config["cfg"].trigger_mode = "auto"
        # sk- Key 形态不会被清洗正则移除，隐私过滤命中 → ori_question 省略
        ws.set_last_user_input("报错 sk-abcdefghijklmnop123456 怎么处理")
        captured = self._capture_payload(monkeypatch, patched_config)
        ws.execute_web_search({"query": "generic technical question"})
        assert "ori_question" not in captured["payload"]


# ======================== 成功路径与双通道（spec 10/11） ========================

class TestSearchSuccess:
    def test_llm_and_display_consistent(self, monkeypatch, patched_config):
        items = [_api_result(title=f"结果{i}", url=f"https://e.com/{i}")
                 for i in range(1, 4)]
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"results": items}))
        out = ws.execute_web_search({"query": "测试查询"})
        text = out["llm_text"]
        display = out["display"]
        assert text.startswith("[web_search]")
        assert "测试查询" in text
        assert len(display["results"]) == 3
        assert display["error"] is False
        # 双通道一致：条数一致、来源一致
        for i, r in enumerate(display["results"], 1):
            assert r["url"] == f"https://e.com/{i}"
            assert f"URL: https://e.com/{i}" in text

    def test_empty_results_notice(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"results": []}))
        out = ws.execute_web_search({"query": "不存在的词"})
        assert "未返回任何结果" in out["llm_text"]
        assert out["display"]["results"] == []
        assert "无结果" in out["display"]["notice"]

    def test_token_budget_truncation(self, monkeypatch, patched_config):
        patched_config["cfg"].token_budget = 100   # 字符预算 500 之内装不下 5 条长摘要
        items = [_api_result(title=f"标题{i}", url=f"https://e.com/{i}",
                             summary="很长的摘要内容" * 30)
                 for i in range(1, 6)]
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"results": items}))
        out = ws.execute_web_search({"query": "预算测试"})
        assert "仅注入前" in out["llm_text"]
        assert "token 预算" in out["display"]["notice"]

    def test_summary_preferred_over_snippet(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(
                                200, {"results": [_api_result(summary="模型阅读摘要")]}))
        out = ws.execute_web_search({"query": "q"})
        assert "模型阅读摘要" in out["llm_text"]


# ======================== 错误矩阵（spec 12） ========================

class TestErrorMatrix:
    def _mock_status(self, monkeypatch, status, body=None):
        monkeypatch.setattr(
            ws.requests, "post",
            lambda *a, **k: _mk_response(
                status, body or {"error": {"message": "boom", "type": "invalid_request_error",
                                           "param": None, "code": status}}))

    def test_400_no_retry_with_guidance(self, monkeypatch, patched_config):
        self._mock_status(monkeypatch, 400)
        out = ws.execute_web_search({"query": "q"})
        assert "HTTP 400" in out["llm_text"]
        assert "boom" in out["llm_text"]
        assert "修正参数" in out["llm_text"]

    def test_401_no_retry(self, monkeypatch, patched_config):
        self._mock_status(monkeypatch, 401)
        out = ws.execute_web_search({"query": "q"})
        assert "认证失败" in out["llm_text"]

    def test_402_recharge_hint(self, monkeypatch, patched_config):
        self._mock_status(monkeypatch, 402)
        out = ws.execute_web_search({"query": "q"})
        assert "余额不足" in out["llm_text"]
        assert "充值" in out["llm_text"]

    def test_500_retries_then_fails(self, monkeypatch, patched_config):
        patched_config["cfg"].max_retries = 2
        calls = {"n": 0}

        def _flaky(*a, **k):
            calls["n"] += 1
            return _mk_response(500, {"error": {"message": "server error"}})

        monkeypatch.setattr(ws.requests, "post", _flaky)
        out = ws.execute_web_search({"query": "q"})
        assert calls["n"] == 3   # 1 次原始 + 2 次重试
        assert "[ERROR]" in out["llm_text"]

    def test_error_message_extraction(self, monkeypatch, patched_config):
        self._mock_status(monkeypatch, 413, {"error": {"message": "请求体超限",
                                                       "type": "request_too_large",
                                                       "param": None, "code": 413}})
        out = ws.execute_web_search({"query": "q"})
        assert "请求体超限" in out["llm_text"]


# ======================== Fetch：URL 校验（spec 7.3） ========================

class TestFetchUrlGuard:
    def test_ok(self):
        assert ws._check_fetch_url("https://example.com/docs") == ""

    @pytest.mark.parametrize("url,expect_kw", [
        ("file:///C:/Windows/system32/config", "非 http/https"),
        ("ftp://example.com/file", "非 http/https"),
        ("http://localhost/admin", "本地地址"),
        ("http://127.0.0.1:8080/", "内网/本地 IP"),
        ("http://10.1.2.3/x", "内网/本地 IP"),
        ("http://172.16.0.1/x", "内网/本地 IP"),
        ("http://192.168.1.1/", "内网/本地 IP"),
        ("http://169.254.169.254/meta", "内网/本地 IP"),
        ("http://[::1]/", "内网/本地 IP"),
        ("http://intranet.local/page", "本地域名"),
        ("https://user:pass@example.com/", "用户名/密码"),
        ("https://example.com/path?q=" + "a" * 8200, "长度超过"),
        ("notaurl", "非 http/https"),
        ("", "URL 为空"),
    ])
    def test_blocked_urls(self, url, expect_kw):
        reason = ws._check_fetch_url(url)
        assert reason != ""
        assert expect_kw in reason

    def test_sensitive_word_url(self):
        reason = ws._check_fetch_url("https://internal-project.example.com/", ("internal-project",))
        assert "受保护词汇" in reason


# ======================== Fetch：入口与解析 ========================

class TestFetchExecution:
    def test_missing_url(self):
        out = ws.execute_web_fetch({})
        assert out["llm_text"].startswith("[ERROR]")

    def test_no_fetch_api_blocked(self, patched_config):
        patched_config["cfg"].fetch_api = ""
        out = ws.execute_web_fetch({"url": "https://example.com/x"})
        assert out["llm_text"].startswith("[BLOCKED]")
        assert "fetch_api" in out["llm_text"]

    def test_blocked_url_never_hits_network(self, monkeypatch, patched_config):
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            return _mk_response(200, {})

        monkeypatch.setattr(ws.requests, "post", _spy)
        out = ws.execute_web_fetch({"url": "http://192.168.0.1/secret"})
        assert called["n"] == 0
        assert out["llm_text"].startswith("[BLOCKED]")

    def test_success_markdown(self, monkeypatch, patched_config):
        body = {"title": "Example article", "content": "# Example\n\n正文内容。"}
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, body))
        out = ws.execute_web_fetch({"url": "https://example.com/docs"})
        assert "# Example" in out["llm_text"]
        assert out["display"]["title"] == "Example article"
        assert out["display"]["char_count"] == len("# Example\n\n正文内容。")
        assert out["display"]["truncated"] is False

    def test_content_fallback_markdown_text(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"markdown": "# MD"}))
        out = ws.execute_web_fetch({"url": "https://e.com/a"})
        assert "# MD" in out["llm_text"]

    def test_budget_truncation_flag(self, monkeypatch, patched_config):
        patched_config["cfg"].fetch_token_budget = 50   # 字符预算 500（max(500,...) 下限）
        long_md = "x" * 2000
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"content": long_md}))
        out = ws.execute_web_fetch({"url": "https://e.com/long"})
        assert out["display"]["truncated"] is True
        assert "首尾保留注入" in out["llm_text"]
        assert out["display"]["char_count"] == 2000

    def test_403_security_policy(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post", lambda *a, **k: _mk_response(
            403, {"error": {"message": "blocked by policy"}}))
        out = ws.execute_web_fetch({"url": "https://e.com/x"})
        assert "HTTP 403" in out["llm_text"]
        assert "web_search" in out["llm_text"]   # 建议改用 web_search

    def test_404_unreadable_semantics(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post", lambda *a, **k: _mk_response(
            404, {"error": {"message": "no content"}}))
        out = ws.execute_web_fetch({"url": "https://e.com/x"})
        assert "未能获取可用内容" in out["llm_text"]

    def test_missing_content_fields_error_with_raw(self, monkeypatch, patched_config):
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"foo": "bar"}))
        out = ws.execute_web_fetch({"url": "https://e.com/x"})
        assert "[ERROR]" in out["llm_text"]
        assert "500 字符" in out["llm_text"]


# ======================== 基础设施：缓存 / 限流 / 会话计数 ========================

class TestInfrastructure:
    def test_search_cache_hit(self, monkeypatch, patched_config, tmp_path):
        monkeypatch.setattr(
            ws.global_cfg, "base_path",
            SimpleNamespace(logs_root=str(tmp_path)), raising=False)
        calls = {"n": 0}

        def _counting_post(*a, **k):
            calls["n"] += 1
            return _mk_response(200, {"results": [_api_result()]})

        monkeypatch.setattr(ws.requests, "post", _counting_post)
        ws.execute_web_search({"query": "cache test"})
        ws.execute_web_search({"query": "cache test"})
        assert calls["n"] == 1   # 第二次命中缓存
        # 第二次输出带缓存标记
        out2 = ws.execute_web_search({"query": "cache test"})
        assert out2["display"]["cached"] is True
        assert "本地缓存" in out2["llm_text"]

    def test_cache_need_content_dimension(self, monkeypatch, patched_config, tmp_path):
        monkeypatch.setattr(
            ws.global_cfg, "base_path",
            SimpleNamespace(logs_root=str(tmp_path)), raising=False)
        calls = {"n": 0}

        def _counting_post(*a, **k):
            calls["n"] += 1
            return _mk_response(200, {"results": [_api_result()]})

        monkeypatch.setattr(ws.requests, "post", _counting_post)
        patched_config["cfg"].need_content = False
        ws.execute_web_search({"query": "dim"})
        patched_config["cfg"].need_content = True
        ws.execute_web_search({"query": "dim"})
        assert calls["n"] == 2   # need_content 维度不同 → 缓存不命中

    def test_cache_ttl_disabled(self, monkeypatch, patched_config, tmp_path):
        monkeypatch.setattr(
            ws.global_cfg, "base_path",
            SimpleNamespace(logs_root=str(tmp_path)), raising=False)
        patched_config["cfg"].cache_ttl = 0
        calls = {"n": 0}

        def _counting_post(*a, **k):
            calls["n"] += 1
            return _mk_response(200, {"results": [_api_result()]})

        monkeypatch.setattr(ws.requests, "post", _counting_post)
        ws.execute_web_search({"query": "ttl off"})
        ws.execute_web_search({"query": "ttl off"})
        assert calls["n"] == 2   # TTL=0 禁用缓存，两次都真实请求

    def test_cache_expired(self, tmp_path):
        cache = ws.NamespacedKVCache(tmp_path / "c.json", ttl=1)
        cache.put("k", {"v": 1})
        assert cache.get("k") == {"v": 1}
        # 手动把时间戳拨回过期
        cache._data["k"]["ts"] = time.time() - 2
        assert cache.get("k") is None

    def test_fetch_cache_by_url(self, monkeypatch, patched_config, tmp_path):
        monkeypatch.setattr(
            ws.global_cfg, "base_path",
            SimpleNamespace(logs_root=str(tmp_path)), raising=False)
        calls = {"n": 0}

        def _counting_post(*a, **k):
            calls["n"] += 1
            return _mk_response(200, {"title": "T", "content": "# C"})

        monkeypatch.setattr(ws.requests, "post", _counting_post)
        ws.execute_web_fetch({"url": "https://e.com/doc"})
        ws.execute_web_fetch({"url": "https://e.com/doc"})
        assert calls["n"] == 1

    def test_session_limit_search(self, monkeypatch, patched_config):
        patched_config["cfg"].max_calls_per_session = 1
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"results": []}))
        ws.execute_web_search({"query": "first"})     # 第 1 次（空结果）占用配额
        out = ws.execute_web_search({"query": "second"})
        assert out["llm_text"].startswith("[BLOCKED]")
        assert "会话级调用上限" in out["llm_text"]

    def test_session_limit_fetch_separate_counting(self, monkeypatch, patched_config):
        patched_config["cfg"].max_calls_per_session = 1
        monkeypatch.setattr(ws.requests, "post", lambda *a, **k: _mk_response(
            200, {"title": "T", "content": "# C"}))
        out1 = ws.execute_web_fetch({"url": "https://e.com/1"})
        out2 = ws.execute_web_fetch({"url": "https://e.com/2"})
        assert not out1["llm_text"].startswith("[BLOCKED]")
        assert out2["llm_text"].startswith("[BLOCKED]")

    def test_rate_limiter_interval(self):
        limiter = ws.RateLimiter(0.2)
        t0 = time.time()
        limiter.wait()
        limiter.wait()
        assert time.time() - t0 >= 0.2


# ======================== ori_question 构造（spec 9.4） ========================

class TestOriQuestion:
    def test_private_mode_returns_empty(self, cfg_factory):
        ws.set_last_user_input("anything")
        assert ws._build_ori_question(cfg_factory.make(trigger_mode="private")) == ""

    def test_auto_mode_cleans_noise(self, cfg_factory):
        ws.set_last_user_input("查看 D:/a/b/c.md 文件 @ref <file_view> 最新版本")
        q = ws._build_ori_question(cfg_factory.make(trigger_mode="auto"))
        assert "D:" not in q
        assert "@" not in q
        assert "<" not in q
        assert "最新版本" in q

    def test_auto_mode_privacy_omitted(self, cfg_factory):
        ws.set_last_user_input("错误：D:\\x\\y.py 第 3 行崩溃 sk-abcdef123456789")
        assert ws._build_ori_question(cfg_factory.make(trigger_mode="auto")) == ""


# ======================== 隐私过滤（双通道） ========================

class TestPrivacy:
    def test_query_patterns(self):
        assert ws.check_query_privacy("D:\\project 文件") != ""
        assert ws.check_query_privacy("key sk-abcdefghijklmnop") != ""
        assert ws.check_query_privacy("Bearer abc123") != ""
        assert ws.check_query_privacy("Python 3.14 free-threading 稳定性") == ""

    def test_sensitive_words(self):
        assert "受保护词汇" in ws.check_query_privacy("MyProject 问题", ("myproject",))

    def test_ori_question_uses_same_filter(self, cfg_factory):
        # ori_question 附加前同样过隐私过滤（TestOriQuestion.test_auto_mode_privacy_omitted
        # 已覆盖，这里验证空消息场景）
        ws.set_last_user_input("")
        assert ws._build_ori_question(cfg_factory.make(trigger_mode="auto")) == ""


# ======================== CLI 渲染冒烟（spec 14：display -> Rich 面板无异常） ========================

class TestCliRenderSmoke:
    def test_render_search_display(self, monkeypatch):
        from wdclaude.cli import cli_print
        display = {
            "tool": "web_search",
            "query_or_url": "测试查询",
            "provider": "DeepSeek-main",
            "cached": False,
            "results": [{
                "title": "标题", "url": "https://e.com/1", "snippet": "摘要内容",
                "date": "2026-06-01", "site_name": "Example",
                "authority": 3, "rerank_score": 2.4,
            }],
            "notice": "",
            "error": False,
        }
        cli_print.print_web_display(display)   # 不抛异常即可

    def test_render_search_empty(self, monkeypatch):
        from wdclaude.cli import cli_print
        cli_print.print_web_display({
            "tool": "web_search", "query_or_url": "q", "provider": "P",
            "cached": False, "results": [], "notice": "无结果", "error": False})

    def test_render_fetch_display(self, monkeypatch):
        from wdclaude.cli import cli_print
        cli_print.print_web_display({
            "tool": "web_fetch", "query_or_url": "https://e.com/doc",
            "title": "Example", "provider": "DeepSeek-main", "cached": False,
            "char_count": 1234, "truncated": False, "preview": "# Head",
            "notice": "", "error": False})

    def test_render_error_display(self, monkeypatch):
        from wdclaude.cli import cli_print
        cli_print.print_web_display({
            "tool": "web_fetch", "query_or_url": "https://e.com/x",
            "provider": "", "cached": False, "results": [],
            "notice": "[BLOCKED] URL 安全校验未通过：内网/本地 IP", "error": True,
            "title": "", "char_count": 0, "truncated": False, "preview": ""})


# ======================== 标签解析回归（tool_executor 接入） ========================

class TestTagParsing:
    def test_strict_web_fetch(self):
        from wdclaude.llm_tool import tool_executor
        remaining, tools = tool_executor.parse_tools(
            '查一下文档\n<web_fetch url="https://example.com/docs"/>\n结束')
        assert len(tools) == 1
        assert tools[0]["llm_tool"] == "web_fetch"
        assert tools[0]["params"]["url"] == "https://example.com/docs"
        assert "查一下文档" in remaining

    def test_strict_web_search_unchanged(self):
        from wdclaude.llm_tool import tool_executor
        remaining, tools = tool_executor.parse_tools(
            '<web_search query="python asyncio 教程" max_results="3"/>')
        assert tools[0]["llm_tool"] == "web_search"
        assert tools[0]["params"]["query"] == "python asyncio 教程"
        assert tools[0]["params"]["max_results"] == 3

    def test_lenient_web_fetch_unclosed_quote(self):
        from wdclaude.llm_tool import tool_executor
        # 402 现场：属性值引号未闭合的畸形标签，lenient 层应能修复
        remaining, tools = tool_executor.parse_tools(
            '<web_fetch url="https://example.com/docs\n好的我来抓取')
        assert len(tools) == 1
        assert tools[0]["llm_tool"] == "web_fetch"

    def test_web_fetch_in_markdown_code_not_parsed(self):
        from wdclaude.llm_tool import tool_executor
        remaining, tools = tool_executor.parse_tools(
            '示例文档：\n```\n<web_fetch url="https://example.com/x"/>\n```')
        assert tools == []


# ======================== 兼容性：纯字符串旧工具不受影响 ========================

class TestBackwardCompat:
    def test_web_tool_output_shape(self, monkeypatch, patched_config):
        """web 工具返回双通道 dict（含 llm_text + display 两个键）。"""
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"results": [_api_result()]}))
        out = ws.execute_web_search({"query": "shape"})
        assert set(out.keys()) == {"llm_text", "display"}
        assert isinstance(out["llm_text"], str)
        assert isinstance(out["display"], dict)
        json.dumps(out, ensure_ascii=False)   # 可序列化（缓存/日志安全）


# ======================== 首尾保留截断与失败熔断 ========================

class TestTruncateMiddle:
    def test_short_text_untouched(self):
        text, truncated, omitted = ws._truncate_middle("短文本", 500)
        assert text == "短文本" and truncated is False and omitted == 0

    def test_keeps_head_and_tail(self):
        text = "HEAD" + "m" * 3000 + "TAIL"
        out, truncated, omitted = ws._truncate_middle(text, 500)
        assert truncated is True
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")          # 尾部数据表不被丢弃
        assert omitted > 0
        assert len(out) <= 500 + 130          # 预算 + 省略标记

    def test_marker_mentions_omitted_count(self):
        out, _, omitted = ws._truncate_middle("x" * 5000, 600)
        assert f"省略 {omitted} 字符" in out

    def test_token_budget_uses_chars_per_token(self):
        # 预算 1000 token × 2.5 字符 ≈ 2500 字符，明显大于旧实现（×2 = 2000）
        assert int(1000 * ws._CHARS_PER_TOKEN) == 2500


class TestFetchFailureGuards:
    def test_negative_cache_prevents_second_request(self, monkeypatch, patched_config):
        calls = {"n": 0}

        def _fail(*a, **k):
            calls["n"] += 1
            raise ws.requests.exceptions.Timeout("boom")

        monkeypatch.setattr(ws.requests, "post", _fail)
        patched_config["cfg"].max_retries = 0
        out1 = ws.execute_web_fetch({"url": "https://slow.example.com/a"})
        out2 = ws.execute_web_fetch({"url": "https://slow.example.com/a"})
        assert calls["n"] == 1                       # 第二次走负面缓存，未再请求
        assert out1["llm_text"].startswith("[ERROR]")
        assert "近期已抓取失败" in out2["llm_text"]

    def test_domain_breaker_opens_after_threshold(self, monkeypatch, patched_config):
        calls = {"n": 0}

        def _fail(*a, **k):
            calls["n"] += 1
            raise ws.requests.exceptions.Timeout("boom")

        monkeypatch.setattr(ws.requests, "post", _fail)
        patched_config["cfg"].max_retries = 0
        patched_config["cfg"].domain_fail_threshold = 2
        patched_config["cfg"].fetch_fail_ttl = 0     # 关掉负面缓存，单独验证熔断
        ws.execute_web_fetch({"url": "https://bad.example.com/1"})
        ws.execute_web_fetch({"url": "https://bad.example.com/2"})
        out = ws.execute_web_fetch({"url": "https://bad.example.com/3"})
        assert calls["n"] == 2                       # 第 3 个 URL 被熔断跳过
        assert "域名熔断中" in out["llm_text"]

    def test_breaker_reset_on_new_session(self, monkeypatch, patched_config):
        ws._get_breaker(patched_config["cfg"]).record_fail("https://x.com/a")
        ws.reset_session_counters()
        assert ws._breaker_instance is None

    def test_success_clears_domain_failure(self, monkeypatch, patched_config):
        breaker = ws._get_breaker(patched_config["cfg"])
        breaker.record_fail("https://e.com/a")
        assert breaker._fails.get("e.com") == 1
        monkeypatch.setattr(ws.requests, "post",
                            lambda *a, **k: _mk_response(200, {"content": "# ok"}))
        ws.execute_web_fetch({"url": "https://e.com/b"})
        assert "e.com" not in breaker._fails
