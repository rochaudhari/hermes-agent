"""Primary OpenAI wire-client kwargs, shared by the sync builder and its ``AsyncOpenAI`` twin.

``agent_runtime_helpers.create_openai_client`` builds the sync client every primary path uses;
``create_async_openai_client`` builds the async client a Relay-managed chat stream reads on its
agent's own loop (``relay_llm_agent_loop``). Both run the same kwargs phases below, so the async
client cannot drift from the sync one: credential, configured headers, TLS verify and proxies are
exactly the class of parity bug a rebuild-from-a-snapshot shipped before (#109595, #109837).
"""

from __future__ import annotations

import weakref
from typing import Any

# Sync clients the plain wire rung built on Hermes' own transport: only these have an async twin
# rebuilt from the same kwargs. Provider-supplied, Gemini-native, MoA and Bedrock-signed clients and
# injected test doubles never land here.
_TWINNABLE: "weakref.WeakSet[Any]" = weakref.WeakSet()


def has_async_twin(client: Any) -> bool:
    return client in _TWINNABLE


def normalized_client_kwargs(agent, client_kwargs: dict) -> dict:
    """A private copy of ``client_kwargs`` with the OpenAI base URL and provider-profile extras."""
    from agent import agent_runtime_helpers as helpers
    from agent.auxiliary_client import _to_openai_base_url
    # Treat client_kwargs as read-only: callers pass agent._client_kwargs (or shallow copies of it),
    # and any in-place mutation leaks back into the stored dict and is reused on later requests.
    # #10933 hit this by injecting an httpx.Client transport that was torn down after the first
    # request, so the next request wrapped a closed transport and raised "Cannot send a request, as
    # the client has been closed" on every retry. The copy locks the contract so future
    # transport/keepalive work can't reintroduce the same class of bug.
    client_kwargs = dict(client_kwargs)
    if client_kwargs.get("base_url"):
        client_kwargs["base_url"] = _to_openai_base_url(client_kwargs["base_url"])
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(getattr(agent, "provider", ""))
        if profile is not None:
            for key, value in profile.build_client_kwargs_extras(
                base_url=client_kwargs.get("base_url", "")
            ).items():
                client_kwargs.setdefault(key, value)
    except Exception:  # health: allow BLE001 -- moved verbatim from create_openai_client; a broken profile hook must not block the client
        helpers._ra().logger.debug("Provider client-kwargs hook skipped", exc_info=True)
    return client_kwargs


def pop_httpx_verify(client_kwargs: dict) -> Any:
    """Pop the TLS settings off ``client_kwargs`` (validating proxies and base URL) and return the
    resolved httpx ``verify`` value the client's transport must carry."""
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)
    httpx_verify = resolve_httpx_verify(
        ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg,
        base_url=str(client_kwargs.get("base_url", "")),
    )
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    return httpx_verify


def finish_wire_client_kwargs(client_kwargs: dict) -> None:
    """Route signing, retry policy and required headers every OpenAI-wire client carries."""
    from agent import agent_runtime_helpers as helpers
    # Bedrock Mantle: the ``aws-sdk`` placeholder is a sentinel for IAM-chain auth, not a bearer token.
    # Every rebuild from bare ``{api_key, base_url}`` kwargs (switch_model, fallback restore, credential
    # rotation, request-scoped clients) must reinstall the SigV4 http_client or Mantle answers 401.
    if "bedrock-mantle." in str(client_kwargs.get("base_url") or ""):
        from agent.bedrock_adapter import configure_bedrock_openai_client_kwargs
        timeout = client_kwargs.get("timeout")
        configure_bedrock_openai_client_kwargs(
            client_kwargs, timeout=timeout if isinstance(timeout, (int, float)) else None,
        )
    # Delegate all rate-limit / 5xx retry to hermes's outer conversation loop, which honors Retry-After and
    # applies adaptive/jittered backoff. The OpenAI SDK default (max_retries=2) uses its own 1-2s backoff
    # that ignores Retry-After and double-retries inside our loop — the same deadlock the Anthropic clients
    # hit (#26293). This is the single chokepoint every primary OpenAI/aggregator client passes through
    # (init, switch_model, recovery, restore, request-scoped); auxiliary_client builds its own clients and
    # keeps SDK retries because it is NOT wrapped by the conversation loop.
    client_kwargs.setdefault("max_retries", 0)
    helpers._ensure_copilot_headers(client_kwargs)
    # All primary construction and recovery paths must identify Hermes to the official Codex
    # endpoint, including snapshots with custom header overrides.
    from agent.codex_headers import apply_required_codex_headers
    apply_required_codex_headers(
        client_kwargs, access_token=client_kwargs.get("api_key", ""),
        base_url=str(client_kwargs.get("base_url", "")),
    )


def build_wire_client(agent, client_kwargs: dict, httpx_verify: Any, *, reason: str, shared: bool) -> Any:
    """The build ladder's plain OpenAI-wire rung (sync). A client on Hermes' own keepalive transport is
    recorded for ``has_async_twin``; a route bringing its own transport (Bedrock Mantle's SigV4
    signer) is not."""
    from agent import agent_runtime_helpers as helpers
    finish_wire_client_kwargs(client_kwargs)
    own_transport = "http_client" not in client_kwargs
    # TCP keepalives so dead provider connections are detected (~60s) instead of hanging in
    # CLOSE-WAIT. Injected into the local copy only, so each client gets its own httpx.Client;
    # pinned by tests/agent/test_create_openai_client_reuse.py. What IS shared across those per-client wrappers is the
    # connection pool: ``build_keepalive_http_client`` mounts a process-shared ``HTTPTransport``
    # behind a per-client view whose ``close()`` is a no-op for the pool, so a closed wrapper
    # never takes a sibling's (or the successor's) connections with it
    # (tests/agent/test_shared_http_transport.py).
    # Without this, a peer that drops mid-stream leaves the socket in a state where epoll_wait never fires,
    # ``httpx`` read timeout may not trigger, and the agent hangs until manually killed. Probes after 30s
    # idle, retry every 10s, give up after 3 → dead peer detected within ~60s. Safety against #10933: the
    # copy ``normalized_client_kwargs`` made means this injection only lands in the local per-call
    # copy, never back into ``agent._client_kwargs``. Each ``_create_openai_client`` invocation therefore
    # gets its OWN fresh ``httpx.Client`` whose lifetime is tied to the OpenAI client it is passed to. When
    # the OpenAI client is closed (rebuild, teardown, credential rotation), the paired ``httpx.Client``
    # closes with it, and the next call constructs a fresh one — no stale closed transport can be reused.
    if own_transport:
        keepalive_http = agent._build_keepalive_http_client(client_kwargs.get("base_url", ""), verify=httpx_verify)
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # ``process_bootstrap.OpenAI`` is a lazy SDK proxy; resolved at call time so tests can patch it.
    from agent import process_bootstrap
    from openai import OpenAI
    client = process_bootstrap.OpenAI(**client_kwargs)
    # Routing proxies name the deployment they served in a response header (#54864).
    from agent.served_model import install_served_model_capture
    install_served_model_capture(agent, client)
    helpers._ra().logger.info("OpenAI client created (%s, shared=%s) %s", reason, shared, agent._client_log_context())
    if own_transport and isinstance(client, OpenAI):
        _TWINNABLE.add(client)
    return client


def create_async_openai_client(agent, client_kwargs: dict, *, reason: str) -> Any:
    """The ``AsyncOpenAI`` twin of a ``has_async_twin`` client, from the kwargs it was built from, for
    the calling thread's running loop (its httpcore pool binds to the loop that first uses it)."""
    from agent import agent_runtime_helpers as helpers
    from agent.auxiliary_async_rebuild import async_credential
    from agent.process_bootstrap import build_keepalive_http_client
    from agent.ssl_verify import platform_ssl_context
    client_kwargs = normalized_client_kwargs(agent, client_kwargs)
    httpx_verify = pop_httpx_verify(client_kwargs)
    finish_wire_client_kwargs(client_kwargs)
    client_kwargs["api_key"] = async_credential(client_kwargs.get("api_key"))
    # The shared platform context: a default ``verify=True`` loads three fresh SSL contexts per client
    # (~14 ms, ~2 MB pinned for the agent's lifetime).
    http_client = build_keepalive_http_client(str(client_kwargs.get("base_url", "")), async_mode=True,
                                              verify=platform_ssl_context() if httpx_verify is True else httpx_verify)
    if http_client is not None:
        client_kwargs["http_client"] = http_client
    from openai import AsyncOpenAI
    client = AsyncOpenAI(**client_kwargs)
    from agent.served_model import install_served_model_capture
    install_served_model_capture(agent, client)
    helpers._ra().logger.info("Async OpenAI client created (%s) %s", reason, agent._client_log_context())
    return client
