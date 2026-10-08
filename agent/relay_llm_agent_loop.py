"""Relay-managed chat streams read on their agent's own long-lived event loop.

A managed stream on a private loop reads each provider chunk with ``asyncio.to_thread(next, ...)``:
an executor hop (two cross-thread wake-ups) per chunk and a sync generator a close can race from
the loop thread (#132048). Here each agent owns ONE loop: Relay's pipeline and an ``AsyncOpenAI``
request client both run on it, so the worker thread driving the stream reads the socket itself.
The loop and its client outlive streams; each stream gets a fresh default executor, shut down when
it releases the loop, so no pool thread idles for the agent's lifetime.

Relay's own bridge is unchanged: it still schedules every provider ``__anext__`` onto the loop and
completes every output chunk from its tokio threads via ``call_soon_threadsafe``, so each chunk
still crosses threads and still depends on the loop's self-pipe wake-up (#132606).

Ownership follows the request-slot rules in ``client_lifecycle.py``: a stranger thread only shuts
the client's sockets down, and every close happens ON the loop while no stream holds it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class AgentStreamLoop:
    """One agent's long-lived loop and the async OpenAI client bound to it. One stream at a time holds
    the loop; a concurrent second stream gets None and keeps the private-loop path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._busy = self._retired = False
        self._client: Any = None
        self._client_kwargs: dict | None = None
        self._retiring: list[Any] = []  # aborted or superseded clients: closed with the loop on release
        self._executor: ThreadPoolExecutor | None = None

    def acquire(self) -> asyncio.AbstractEventLoop | None:
        with self._lock:
            if self._busy:
                return None
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            self._busy = True
            # The loop's pool runs getaddrinfo and token providers; one started here would otherwise idle
            # for the agent's lifetime (128 cached gateway agents = 128 threads). Released with the stream.
            self._executor = ThreadPoolExecutor(thread_name_prefix="relay-agent-loop")
            self._loop.set_default_executor(self._executor)
            return self._loop

    def runs_current_loop(self) -> bool:
        """True inside the stream holding this loop (only a holding stream runs it)."""
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def checkout(self, agent: Any, request_kwargs: dict, *, reason: str) -> Any:
        """The async twin for ``request_kwargs``, built on first use and whenever the kwargs change or an
        abort retired the last one. Runs on the loop, inside the stream holding it."""
        with self._lock:
            if self._client is not None and self._client_kwargs != request_kwargs:
                self._retiring.append(self._client)
                self._client = None
            if self._client is not None:
                return self._client
        from agent.agent_runtime_helpers_openai_client import create_async_openai_client
        client = create_async_openai_client(agent, request_kwargs, reason=reason)
        with self._lock:
            self._client, self._client_kwargs = client, request_kwargs
        return client

    def abort(self, client: Any) -> int:
        """Stranger-thread abort of an in-flight request: ``shutdown()`` its sockets (FD-safe from any
        thread; the loop's reader unwinds) and retire the client so no later request reuses it.
        Returns the sockets shut down."""
        with self._lock:
            if client is self._client:
                self._retiring.append(client)
                self._client = None
        from agent.agent_runtime_helpers import force_close_tcp_sockets
        count = force_close_tcp_sockets(client)
        logger.info("Agent-loop OpenAI client aborted (tcp_force_closed=%d)", count)
        return count

    def abort_in_flight(self) -> int:
        """Abort the holding stream's request, if any (the abandoned-worker drain); sockets shut down."""
        with self._lock:
            client = self._client if self._busy else None
        return 0 if client is None else self.abort(client)

    def release(self, loop: asyncio.AbstractEventLoop, *, abandoned: bool) -> None:
        """Hand the loop back after its stream closed; retiring clients, leftover tasks or a pending
        teardown retire it too. ``abandoned``: a timed-out close left it running on a daemon thread,
        so it and its clients are dropped to GC, never closed under that thread."""
        with self._lock:
            executor, self._executor = self._executor, None
            retire = abandoned or self._retired or bool(self._retiring) or bool(asyncio.all_tasks(loop))
            clients = self._detach() if retire else []
        try:
            if retire and not abandoned:
                _shutdown(loop, clients)
        finally:
            if not abandoned:  # an abandoned loop keeps its pool, like a private one
                executor.shutdown(wait=False)
            with self._lock:
                self._busy = False

    def close(self) -> None:
        """Teardown from any thread: close an idle loop and its client now; abort a busy loop's
        in-flight request and retire the loop when its stream releases it."""
        with self._lock:
            if self._busy:
                self._retired = True
                in_flight, loop, clients = self._client, None, []
            else:
                in_flight, loop, clients = None, self._loop, self._detach()
        if in_flight is not None:
            self.abort(in_flight)
        if loop is not None:
            _shutdown(loop, clients)

    def _detach(self) -> list[Any]:
        """Under ``_lock``: forget the loop and return every client it owns, for ``_shutdown``."""
        clients = [*self._retiring, *([self._client] if self._client is not None else [])]
        self._loop = self._client = self._client_kwargs = None
        self._retiring, self._retired = [], False
        return clients


class AgentLoopRequestClient:
    """One request's view of the agent loop's async client: ``chat`` for the stream, and ``cancel()``,
    the abort ``_abort_request_slot_client`` calls on a client that is not a socket pool."""

    def __init__(self, owner: AgentStreamLoop, client: Any) -> None:
        self._owner, self._client = owner, client

    @property
    def chat(self) -> Any:
        return self._client.chat

    def cancel(self) -> None:
        self._owner.abort(self._client)


def agent_stream_loop(agent: Any) -> AgentStreamLoop:
    """The agent's loop holder, created on first use (tests build agents via ``AIAgent.__new__``)."""
    with agent._openai_client_lock():
        holder = getattr(agent, "_relay_agent_loop", None)
        if holder is None:
            holder = agent._relay_agent_loop = AgentStreamLoop()
        return holder


def chat_stream_request_client(agent: Any, *, reason: str, api_kwargs: dict) -> Any:
    """Request client for one chat stream: on the agent's loop, the request slot's own client is swapped
    for its async twin (``has_async_twin``: the build ladder's plain wire rung made it); any other
    client (provider-supplied, Gemini-native, MoA, Bedrock-signed, injected, untracked) serves as is."""
    from agent.agent_runtime_helpers_openai_client import has_async_twin
    client = agent._create_request_openai_client(reason=reason, api_kwargs=api_kwargs)
    holder = getattr(agent, "_relay_agent_loop", None)
    if holder is None or not has_async_twin(client) or not holder.runs_current_loop():
        return client
    request_kwargs = agent._request_slot_kwargs(client)
    if request_kwargs is None:
        return client
    try:
        async_client = holder.checkout(agent, request_kwargs, reason=reason)
    except Exception:  # health: allow BLE001 -- the async twin is an optimisation; failing to build it must not fail a request the sync client serves
        logger.warning("Agent-loop async OpenAI client unavailable; streaming on the sync request client", exc_info=True)
        return client
    agent._close_request_openai_client(client, reason="stream_request_complete")  # unused: back to its slot
    return AgentLoopRequestClient(holder, async_client)


def close_agent_stream_loop(agent: Any, *, reason: str) -> None:
    """Teardown hook (``release_clients``/``close``): close the agent's loop and its async client."""
    holder = getattr(agent, "_relay_agent_loop", None)
    if holder is not None:
        logger.debug("Closing the agent's Relay stream loop (%s)", reason)
        holder.close()


def _shutdown(loop: asyncio.AbstractEventLoop, clients: list[Any]) -> None:
    """Cancel leftover tasks, close ``clients`` and async generators on the idle ``loop``, then close it.
    A close that outlives ``relay_llm._ACLOSE_TIMEOUT`` leaves the loop running on its daemon thread."""
    from agent import relay_llm

    async def settle(*, cancel: bool) -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending if cancel else ():
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=relay_llm._ACLOSE_TIMEOUT / 4)

    async def finish() -> None:
        await settle(cancel=True)
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.debug("Agent-loop OpenAI client close failed", exc_info=True)
        await loop.shutdown_asyncgens()
        # A transport generator collected while closing scheduled its aclose(); run it before close()
        # drops it unawaited.
        await asyncio.sleep(0)
        await settle(cancel=False)

    try:
        completed = relay_llm._run_on_loop_bounded(
            loop, finish, name="relay-agent-loop-close", what="Relay agent loop close")
    except Exception:
        logger.debug("Relay agent loop close failed", exc_info=True)
        completed = True
    if completed:
        loop.close()
