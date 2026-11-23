from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tui.common import bench


class _Fake(BaseHTTPRequestHandler):
    seen: list[str] = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def log_message(self, *_a):
        pass

    def do_GET(self):
        raw = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.seen.append(self.path)
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
        try:
            if self.path == "/v1/chat/completions":
                assert body["stream"] is True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(0.05)
                for word in ("a", "b", "c", "d"):
                    chunk = {"choices": [{"delta": {"content": word}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.01)
                usage = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}}
                self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
                return
            if self.path == "/v1/embeddings":
                payload = {"data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 7}}
            elif self.path == "/v1/rerank":
                assert body["documents"]
                payload = {"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"prompt_tokens": 9}}
            else:
                self.send_response(404)
                self.end_headers()
                return
            time.sleep(0.02)
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        finally:
            with cls.lock:
                cls.active -= 1


@pytest.fixture
def server():
    _Fake.seen, _Fake.active, _Fake.peak = [], 0, 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


@pytest.mark.parametrize(
    "backend, params, expected",
    [
        ("llamacpp", {"embedding": True, "pooling": "last"}, "embedding"),
        ("llamacpp", {"embeddings": True, "pooling": "rank"}, "rerank"),
        ("llamacpp", {"reranking": True}, "rerank"),
        ("llamacpp", {"ctx-size": 1}, "chat"),
        ("vllm", {"task": "embed"}, "embedding"),
        ("vllm", {"runner": "pooling", "convert": "embed"}, "embedding"),
        ("vllm", {"task": "score"}, "rerank"),
        ("vllm", {}, "chat"),
    ],
)
def test_bench_kind_follows_the_serving_flags(backend, params, expected) -> None:
    assert bench.detect_kind(backend, params) == expected


@pytest.mark.asyncio
async def test_concurrent_chat_load_reports_ttft_and_token_latency(server) -> None:
    result = await bench.run_load(server, "m", kind="chat", concurrency=4, requests=8, warmup=0)
    assert _Fake.peak == 4
    assert result["succeeded"] == 8 and result["failed"] == 0
    assert result["output_tokens"] == 32
    assert result["output_tok_s"] > 0 and result["req_s"] > 0
    assert 0.04 < result["ttft_s"]["p50"] < 1.0
    assert 0 < result["tpot_s"]["p50"] < 0.5
    assert set(result["e2e_s"]) == {"p50", "p90", "p99"}


@pytest.mark.asyncio
async def test_embedding_and_rerank_hit_their_endpoints(server) -> None:
    emb = await bench.run_load(server, "m", kind="embedding", concurrency=2, requests=3, warmup=1)
    rr = await bench.run_load(server, "m", kind="rerank", concurrency=1, requests=2, warmup=0)
    assert _Fake.seen.count("/v1/embeddings") == 4  # warmup + measured
    assert _Fake.seen.count("/v1/rerank") == 2
    assert emb["input_tokens"] == 21 and emb["ttft_s"] is None
    assert rr["succeeded"] == 2


@pytest.mark.asyncio
async def test_failed_requests_are_counted_not_raised() -> None:
    result = await bench.run_load(9, "m", kind="chat", concurrency=1, requests=2, warmup=0)
    assert result["succeeded"] == 0 and result["failed"] == 2
    assert result["errors"] and result["ttft_s"] is None


@pytest.mark.asyncio
async def test_bad_load_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        await bench.run_load(1, "m", kind="chat", concurrency=0, requests=1)
    with pytest.raises(ValueError, match="kind"):
        await bench.run_load(1, "m", kind="nope", concurrency=1, requests=1)


def test_cli_bench_picks_the_kind_and_load_options(server, tmp_path) -> None:
    from typer.testing import CliRunner

    from tui.backends.llamacpp import backend as lbackend
    from tui.cli import app
    from tui.common import profile_store
    from unittest.mock import patch

    profile_store.create_profile(profile_store.StoredProfile(
        name="emb", backend="llamacpp", port=server, hf_repo="o/r", hf_file="m.gguf",
    ))
    (tmp_path / "emb.yaml").write_text("alias: emb\nembedding: true\npooling: last\n")
    runner = CliRunner()
    with patch.object(lbackend, "CONFIG_DIR", tmp_path):
        auto = runner.invoke(app, ["bench", "emb", "--requests", "3", "--warmup", "0", "--json"])
        chat = runner.invoke(app, ["bench", "emb", "--kind", "chat", "--concurrency", "2", "--requests", "4", "--json"])
        bad = runner.invoke(app, ["bench", "emb", "--concurrency", "0"])
    assert auto.exit_code == 0, auto.output
    auto_json = json.loads(auto.stdout)
    assert (auto_json["kind"], auto_json["succeeded"], auto_json["ok"], auto_json["code"]) == ("embedding", 3, True, 0)
    chat_json = json.loads(chat.stdout)
    assert (chat_json["kind"], chat_json["concurrency"], chat_json["succeeded"]) == ("chat", 2, 4)
    assert chat_json["ttft_s"]["p50"] > 0
    assert bad.exit_code == 2


def test_load_mode_rule_is_shared_by_cli_and_tui() -> None:
    assert bench.uses_load_mode("chat", None, None) is False
    assert bench.uses_load_mode("chat", 4, None) is True
    assert bench.uses_load_mode("chat", None, 10) is True
    assert bench.uses_load_mode("embedding", None, None) is True


def test_load_summary_line() -> None:
    line = bench.load_summary({
        "kind": "chat", "succeeded": 8, "failed": 0, "req_s": 2.0, "output_tok_s": 150.0, "input_tok_s": 0.0,
        "ttft_s": {"p50": 0.12, "p90": 0.2, "p99": 0.3}, "tpot_s": {"p50": 0.02, "p90": 0.03, "p99": 0.04},
        "e2e_s": {"p50": 1.0, "p90": 1.2, "p99": 1.5}, "concurrency": 4,
    })
    assert "150.0 tok/s" in line and "TTFT p50 120 ms" in line and "c=4" in line


@pytest.mark.asyncio
async def test_bench_modal_returns_the_options() -> None:
    from textual.app import App
    from textual.widgets import Input

    from tui.common.bench_modal import BenchOptionsModal

    results = []
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(BenchOptionsModal("p"), results.append)
        await pilot.pause()
        app.screen.query_one("#bench-concurrency", Input).value = "4"
        app.screen.query_one("#bench-requests", Input).value = "20"
        app.screen.query_one("#bench-run").press()
        await pilot.pause()
        app.push_screen(BenchOptionsModal("p"), results.append)
        await pilot.pause()
        app.screen.query_one("#bench-concurrency", Input).value = "0"
        app.screen.query_one("#bench-run").press()  # invalid → stays open
        await pilot.pause()
        app.screen.query_one("#bench-cancel").press()
        await pilot.pause()
    assert results == [{"kind": "auto", "concurrency": 4, "requests": 20}, None]


def test_partial_failures_fail_the_bench(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from tui.cli import app
    from tui.common import http, profile_store

    profile_store.create_profile(profile_store.StoredProfile(
        name="emb", backend="llamacpp", port=18999, hf_repo="o/r", hf_file="m.gguf",
    ))
    monkeypatch.setattr(http, "list_served_models", AsyncMock(return_value=["m"]))
    monkeypatch.setattr(http, "profile_api_key", AsyncMock(return_value=""))
    stats = {"model": "m", "kind": "chat", "concurrency": 2, "requests": 3, "succeeded": 2, "failed": 1, "errors": ["boom"]}
    monkeypatch.setattr(bench, "run_load", AsyncMock(return_value=stats))
    result = CliRunner().invoke(app, ["bench", "emb", "--kind", "chat", "--concurrency", "2", "--json"])
    record = json.loads(result.stdout)
    assert result.exit_code == 1
    assert (record["ok"], record["code"], record["succeeded"], record["failed"]) == (False, 1, 2, 1)



def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _BadStream(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if "error" in self.path:
            self.wfile.write(b'data: {"error": {"message": "context overflow"}}\n\n')
        # no [DONE]: the stream just stops


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/error", "/truncated"])
async def test_broken_streams_count_as_failures(monkeypatch, path) -> None:
    server = _serve(_BadStream)
    try:
        monkeypatch.setitem(bench._PATHS, "chat", path)
        result = await bench.run_load(server.server_address[1], "m", kind="chat", concurrency=1, requests=1, warmup=0)
    finally:
        server.shutdown()
    assert (result["succeeded"], result["failed"]) == (0, 1), result["errors"]


class _Slow(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        time.sleep(3)
        self.send_response(500)
        self.end_headers()


@pytest.mark.asyncio
async def test_cancel_does_not_block_the_event_loop() -> None:
    import asyncio

    server = _serve(_Slow)
    try:
        task = asyncio.create_task(bench.run_load(server.server_address[1], "m", kind="embedding", concurrency=2, requests=2, warmup=0))
        await asyncio.sleep(0.3)
        started = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.perf_counter() - started < 1.0
    finally:
        server.shutdown()
