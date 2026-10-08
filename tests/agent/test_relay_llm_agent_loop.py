"""Relay-managed chat streams on the agent's own long-lived loop (agent/relay_llm_agent_loop.py).

The stream-level contracts drive ``relay_llm.stream`` directly; the agent-level ones run a real
AIAgent under real Relay managed execution against a local OpenAI-compatible SSE server and assert
what the provider sees: which client asked (the SDK user agent), with which credential, and when
the socket went away.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime
from agent.relay_llm_agent_loop import AgentStreamLoop

CHUNKS = [{"delta": "a"}, {"delta": "b"}, {"delta": "c"}]
REPLY = "Hello from the loop"


class _AsyncChunks:
    """Async provider stream stand-in that records the loop it was read on and its close()."""

    def __init__(self, chunks: list) -> None:
        self._chunks = list(chunks)
        self.loop = asyncio.get_running_loop()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def relay_session(tmp_path, monkeypatch):
    """A Relay turn on ``session-1`` with managed execution on."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="session-1", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.relay_agent_loop")
    try:
        yield lease.host.relay
    finally:
        lease.host.release_managed_execution("test.relay_agent_loop")
        coordinator.end_turn(turn, outcome="success")
        coordinator.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _stream(factory, holder):
    return relay_llm.stream(
        {"model": "test-model", "messages": []}, factory, session_id="session-1", name="test-provider",
        model_name="test-model", finalizer=lambda: {"content": "complete"}, metadata={"api_mode": "custom"},
        agent_loop=holder,
    )


def _async_opener(opened: list):
    """Stream factory opening an ``_AsyncChunks`` (recorded in ``opened``) on the stream's loop."""

    async def open_stream():
        opened.append(_AsyncChunks(CHUNKS))
        return opened[-1]

    return lambda _request: open_stream()


@pytest.mark.usefixtures("relay_session")
def test_streams_share_the_agent_loop_and_a_busy_one_never_blocks_the_next():
    holder, opened = AgentStreamLoop(), []

    first = _stream(_async_opener(opened), holder)
    assert next(first) == CHUNKS[0]
    # While ``first`` holds the agent's loop, a sync stream runs on a private loop of its own.
    assert list(_stream(lambda _request: iter(CHUNKS), holder)) == CHUNKS
    assert next(first) == CHUNKS[1]
    first.close()
    assert opened[0].closed  # an early stop awaits the async stream's own close

    assert list(_stream(_async_opener(opened), holder)) == CHUNKS
    assert opened[1].loop is opened[0].loop  # the loop outlives streams
    holder.close()
    assert opened[0].loop.is_closed()


def test_a_relay_failure_after_the_provider_finished_hands_the_loop_back(relay_session, monkeypatch):
    holder, opened = AgentStreamLoop(), []

    async def fail_after_provider(_name, request, callback, observe_chunk, finalizer, **_kwargs):
        async def generate():
            async for chunk in callback(request):
                observe_chunk(chunk)
            finalizer()
            raise RuntimeError("simulated Relay post-processing failure")
            yield  # pragma: no cover

        return generate()

    with monkeypatch.context() as patch:
        patch.setattr(relay_session.llm, "stream_execute", fail_after_provider)
        assert list(_stream(_async_opener(opened), holder)) == CHUNKS  # served from the provider

    assert list(_stream(_async_opener(opened), holder)) == CHUNKS
    assert opened[1].loop is opened[0].loop and not opened[0].loop.is_closed()
    holder.close()


class _Provider:
    """Local SSE chat-completions server; ``mode`` shapes the next responses."""

    def __init__(self) -> None:
        self.requests: list[dict] = []  # each chat stream's request headers
        self.mode = "stream"  # "stream" | "stall_before_headers" | "stall_after_first"
        self.stalled, self.disconnected = threading.Event(), threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def user_agents(self) -> list[str]:
        return [headers["user-agent"].split("/")[0] for headers in self.requests]

    def _handler(self):
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                if not body.get("stream"):  # the agent's own startup probes are not chat streams
                    self.send_error(404)
                    return
                provider.requests.append({key.lower(): value for key, value in self.headers.items()})
                if provider.mode == "stall_before_headers":
                    self._wait_for_disconnect()
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                for text in ("Hello", " from", " the loop"):
                    self._chunk({"content": text}, None)
                    if provider.mode == "stall_after_first":
                        self._wait_for_disconnect()
                        return
                self._chunk({}, "stop")
                self._write(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")

            def _chunk(self, delta: dict, finish_reason) -> None:
                body = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test/model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
                self._write(f"data: {json.dumps(body)}\n\n".encode())

            def _write(self, data: bytes) -> None:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

            def _wait_for_disconnect(self) -> None:
                self.close_connection = True
                self.connection.settimeout(15)
                provider.stalled.set()
                try:
                    while self.connection.recv(4096):
                        pass
                except TimeoutError:
                    return
                except OSError:
                    pass
                provider.disconnected.set()

        return Handler

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def provider():
    server = _Provider()
    yield server
    server.close()


@pytest.fixture()
def agent(provider, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    relay_runtime._reset_for_tests()
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url=provider.base_url, provider="custom", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent.session_id = "agent-loop-session"
    try:
        yield agent
    finally:
        agent.close()
        relay_runtime._reset_for_tests()


def _managed_call(agent) -> tuple[threading.Thread, dict]:
    """One streaming call on a thread of its own under a Relay turn with managed execution on."""
    outcome: dict = {}

    def run() -> None:
        coordinator = relay_runtime.SESSION_COORDINATOR
        lease = coordinator.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(), session_id=agent.session_id, platform="cli")
        turn = coordinator.begin_turn(lease, turn_id=f"turn-{time.monotonic_ns()}", task_id="task-1")
        lease.host.retain_managed_execution("test.relay_agent_loop")
        try:
            response = agent._interruptible_streaming_api_call(
                {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]})
            outcome["content"] = response.choices[0].message.content
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            lease.host.release_managed_execution("test.relay_agent_loop")
            coordinator.end_turn(turn, outcome="success")
            coordinator.release_conversation(lease)

    thread = threading.Thread(target=run, name="managed-call", daemon=True)
    thread.start()
    return thread, outcome


def _complete_call(agent) -> dict:
    thread, outcome = _managed_call(agent)
    thread.join(30)
    assert not thread.is_alive()
    return outcome


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_a_failed_async_client_build_streams_on_the_sync_client(agent, provider, monkeypatch):
    # The async twin is an optimisation: a build failure (a CA bundle removed after startup) must not
    # fail the turn, and the sync client checked out for the swap must go back to its slot, so the
    # next call swaps again instead of building untracked sync clients forever.
    from agent import agent_runtime_helpers_openai_client as builders

    build, attempts = builders.create_async_openai_client, []

    def build_failing_first(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("ssl_ca_cert bundle is gone")
        return build(*args, **kwargs)

    monkeypatch.setattr(builders, "create_async_openai_client", build_failing_first)

    assert _complete_call(agent).get("content") == REPLY
    assert _complete_call(agent).get("content") == REPLY
    assert provider.user_agents() == ["OpenAI", "AsyncOpenAI"]


def test_no_pool_thread_outlives_an_agent_loop_stream(agent, provider):
    # The loop lives as long as the agent (a gateway caches up to 128); a pool thread started for one
    # stream's hostname lookup must not.
    agent._client_kwargs = {**agent._client_kwargs, "base_url": provider.base_url.replace("127.0.0.1", "localhost")}
    agent.client = agent._create_openai_client(agent._client_kwargs, reason="test", shared=True)

    assert _complete_call(agent).get("content") == REPLY

    assert provider.user_agents() == ["AsyncOpenAI"]
    assert _wait_until(lambda: not [
        thread for thread in threading.enumerate() if thread.name.startswith(("relay-agent-loop", "asyncio_"))])


def test_agent_loop_request_carries_the_sync_clients_credential_and_headers(agent, provider):
    # A rotating token source and a named provider's extra headers live only in the client kwargs (#109595).
    agent._client_kwargs = {**agent._client_kwargs, "api_key": lambda: "rotated-token",
                            "default_headers": {"X-Hermes-Test": "configured"}}
    agent.client = agent._create_openai_client(agent._client_kwargs, reason="test", shared=True)

    assert _complete_call(agent).get("content") == REPLY

    headers = provider.requests[-1]
    assert provider.user_agents() == ["AsyncOpenAI"]
    assert headers["authorization"] == "Bearer rotated-token"
    assert headers["x-hermes-test"] == "configured"


def test_a_provider_supplied_openai_client_keeps_the_sync_stream(agent, provider, monkeypatch):
    # The profile's client may carry a transport the bare kwargs cannot rebuild, even when it is an OpenAI.
    import providers
    from openai import OpenAI
    from providers.base import ProviderProfile

    class _OwnClientProfile(ProviderProfile):
        def create_client(self, **kwargs):
            return OpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"], max_retries=0)

    providers._discover_providers()
    monkeypatch.setitem(providers._REGISTRY, "own-client", _OwnClientProfile(name="own-client"))
    agent.provider = "own-client"
    agent.client = agent._create_openai_client(agent._client_kwargs, reason="test", shared=True)

    assert _complete_call(agent).get("content") == REPLY
    assert provider.user_agents() == ["OpenAI"]


@pytest.mark.parametrize("mode", ["stall_before_headers", "stall_after_first"])
def test_stop_drops_a_silent_agent_loop_request(agent, provider, mode):
    # Before the first byte and mid-body alike, /stop must reach the socket so the provider stops
    # generating into a dropped consumer (#98974); the agent's next stream still works.
    provider.mode = mode
    thread, outcome = _managed_call(agent)
    assert provider.stalled.wait(10)

    agent.interrupt(hard_cancel=True)

    assert provider.disconnected.wait(5)
    thread.join(10)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), InterruptedError)
    assert provider.user_agents() == ["AsyncOpenAI"]
    provider.mode = "stream"
    agent.clear_interrupt()
    assert _complete_call(agent).get("content") == REPLY


def test_release_clients_aborts_the_in_flight_request_and_closes_the_idle_loop(agent, provider):
    provider.mode = "stall_after_first"
    thread, outcome = _managed_call(agent)
    assert provider.stalled.wait(10)

    agent.release_clients()

    assert provider.disconnected.wait(5)
    thread.join(15)
    assert not thread.is_alive()
    assert outcome.get("content") != REPLY
    provider.mode = "stream"
    assert _complete_call(agent).get("content") == REPLY  # a released agent streams on a fresh loop
    assert provider.user_agents() == ["AsyncOpenAI", "AsyncOpenAI"]

    holder = agent._agent_stream_loop()
    loop = holder.acquire()
    holder.release(loop, abandoned=False)
    agent.release_clients()
    assert loop.is_closed()


def test_the_abandoned_worker_drain_reaches_an_agent_loop_stream(agent, provider):
    # A timed-out delegated child is drained from the parent's thread (#94248); the sweep must reach the
    # agent loop's async client, which is what the stream reads, not only the request slot's client.
    provider.mode = "stall_after_first"
    thread, _outcome = _managed_call(agent)
    assert provider.stalled.wait(10)

    agent._drain_transports_after_abandonment(reason="delegate_timeout_immediate")

    assert provider.disconnected.wait(5)
    thread.join(15)
    assert not thread.is_alive()
    provider.mode = "stream"
    assert _complete_call(agent).get("content") == REPLY
    assert provider.user_agents() == ["AsyncOpenAI", "AsyncOpenAI"]
