"""Real-agent load harness for Relay-managed provider streaming.

N concurrent AIAgents (one thread each, like gateway sessions) each run one real streaming call
(``agent._interruptible_streaming_api_call``) against a local OpenAI-compatible SSE server in a
separate process. Every chunk carries the server's send time; the delay is measured where Hermes'
turn loop counts the chunk (``_StreamingCall._count_chunk``, after Relay's pipeline), so it covers
the provider read, the hand-off, Relay and the Hermes callbacks.

Variants (each runs in its own fresh process):
  unmanaged  Relay managed execution off: the SDK stream is iterated directly (Hermes' own floor).
  private    managed on a private loop per stream: Relay pulls each chunk through
             asyncio.to_thread(next, sync_stream) (the agent loop disabled).
  agent_loop managed on the agent's long-lived loop (agent/relay_llm_agent_loop.py): Relay's pipeline
             and the async OpenAI request client both run on it, so the provider read needs no
             executor hop (Relay's own tokio <-> loop hand-offs per chunk remain).

Streams start at random offsets and their chunk gaps jitter (real streams are not aligned).
Each worker runs three phases on the same agents: warm-up (discarded), SHORT chunks, LONG chunks.
CPU and context switches per chunk are MARGINAL: (long - short) / extra chunks, so the per-call
fixed cost (request build, Relay session, finalizer) does not swamp the per-chunk numbers. Delays,
CPU load and chunks delivered come from the long phase. The path column (managed/async-client
calls) proves each variant ran the path it names.

    python evals/relay_stream_handoff/bench.py                       # defaults below
    python evals/relay_stream_handoff/bench.py --ns 1,10,50 --repeat 3 --chunks 100 --gap-ms 20

Context switches need ``resource`` (Linux/macOS); Windows reports n/a.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VARIANTS = ("unmanaged", "private", "agent_loop")

PRODUCER = r'''
import asyncio, json, random, sys, time
chunks, gap, jitter, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rnd = random.Random(seed)
BASE = {"id": "chatcmpl-bench", "object": "chat.completion.chunk", "created": 1, "model": "test/model"}

def frame(payload):
    data = b"data: " + (payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n\n"
    return b"%x\r\n" % len(data) + data + b"\r\n"

async def handle(reader, writer):
    try:
        while True:  # keep-alive: one connection may carry several requests
            head = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            body = await reader.readexactly(length) if length else b"{}"
            count = json.loads(body or b"{}").get("max_tokens") or chunks
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n"
                         b"transfer-encoding: chunked\r\n\r\n")
            await writer.drain()
            await asyncio.sleep(rnd.uniform(0, gap))  # random phase per stream
            for k in range(count):
                delta = {"content": "t%d " % time.monotonic_ns()}  # stamped at send
                if k == 0:
                    delta["role"] = "assistant"
                writer.write(frame({**BASE, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}))
                await writer.drain()
                await asyncio.sleep(gap * rnd.uniform(1 - jitter, 1 + jitter))
            writer.write(frame({**BASE, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            writer.write(frame({**BASE, "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": count, "total_tokens": 10 + count}}))
            writer.write(frame(b"[DONE]") + b"0\r\n\r\n")
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        writer.close()

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 0, backlog=1024)
    print(server.sockets[0].getsockname()[1], flush=True)
    async with server:
        await server.serve_forever()

asyncio.run(main())
'''


def _ctx_switches() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_nvcsw + usage.ru_nivcsw


def run_worker(variant: str, n: int, chunks: int, gap: float, jitter: float, seed: int) -> dict:
    """One variant at one concurrency, in this (fresh) process. Returns the measurements."""
    os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="relay-stream-bench-")
    os.environ["HERMES_STREAM_RETRIES"] = "0"
    sys.path.insert(0, REPO)
    import logging
    logging.disable(logging.WARNING)

    producer = subprocess.Popen([sys.executable, "-c", PRODUCER, str(chunks), str(gap), str(jitter), str(seed)],
                                stdout=subprocess.PIPE, text=True)
    port = int(producer.stdout.readline())
    base_url = f"http://127.0.0.1:{port}/v1"

    from agent import chat_completion_helpers, relay_llm_agent_loop, relay_runtime
    from run_agent import AIAgent

    if variant == "private":
        AIAgent._agent_stream_loop = lambda self: None
    relay_runtime._reset_for_tests()

    delays_us: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    count_chunk = chat_completion_helpers._StreamingCall._count_chunk

    def timed_count_chunk(self, diag, chunk):
        try:
            content = chunk.choices[0].delta.content
        except (AttributeError, IndexError, TypeError):
            content = None
        if record["on"] and content and content.startswith("t"):
            delay = (time.monotonic_ns() - int(content[1:].split(" ", 1)[0])) / 1000
            with lock:
                delays_us.append(delay)
        return count_chunk(self, diag, chunk)

    chat_completion_helpers._StreamingCall._count_chunk = timed_count_chunk

    # Path proof: a variant that silently fell back to another path would make the comparison moot.
    from agent import relay_llm
    paths = {"managed": 0, "async_client": 0}
    start_managed = relay_llm.ManagedLlmStream._start_managed
    handle_init = relay_llm_agent_loop.AgentLoopRequestClient.__init__

    def counted_start_managed(self, attempt, *args):
        with lock:
            paths["managed"] += 1
        start_managed(self, attempt, *args)

    def counted_handle_init(self, owner, client):
        with lock:
            paths["async_client"] += 1
        handle_init(self, owner, client)

    relay_llm.ManagedLlmStream._start_managed = counted_start_managed
    relay_llm_agent_loop.AgentLoopRequestClient.__init__ = counted_handle_init

    agents = []
    for i in range(n):
        agent = AIAgent(api_key="bench-key", base_url=base_url, provider="bench-provider", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.api_mode = "chat_completions"
        agent.session_id = f"relay-bench-{i}"
        agent._interrupt_requested = False
        agents.append(agent)

    retained: list = []
    record = {"on": False}

    def run_one(i: int, agent, tokens: int, start: threading.Barrier) -> None:
        coordinator = relay_runtime.SESSION_COORDINATOR
        lease = coordinator.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(), session_id=agent.session_id, platform="cli")
        turn = coordinator.begin_turn(lease, turn_id=f"bench-turn-{i}-{tokens}", task_id=f"bench-task-{i}")
        if variant != "unmanaged":
            with lock:
                if not retained:
                    lease.host.retain_managed_execution("bench.relay_stream_handoff")
                    retained.append(lease.host)
        start.wait()
        try:
            agent._interruptible_streaming_api_call({"model": "test/model", "max_tokens": tokens,
                                                     "messages": [{"role": "user", "content": "hi"}]})
        except Exception as exc:  # health: allow BLE001 -- benchmark boundary: every error is recorded and reported
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}"[:200])
        finally:
            coordinator.end_turn(turn, outcome="success")
            coordinator.release_conversation(lease)

    def phase(tokens: int, measure_delays: bool) -> dict:
        record["on"] = measure_delays
        start = threading.Barrier(n + 1)
        threads = [threading.Thread(target=run_one, args=(i, a, tokens, start), name=f"bench-stream-{i}")
                   for i, a in enumerate(agents)]
        for t in threads:
            t.start()
        start.wait()
        cs0, cpu0, wall0 = _ctx_switches(), time.process_time(), time.monotonic()
        for t in threads:
            t.join()
        cpu1, cs1, wall1 = time.process_time(), _ctx_switches(), time.monotonic()
        return {"cpu": cpu1 - cpu0, "cs": None if cs0 is None else cs1 - cs0, "wall": wall1 - wall0}

    short = max(5, chunks // 10)
    phase(short, False)  # warm-up: lazy imports, clients, Relay runtime
    p_short = phase(short, False)
    p_long = phase(chunks, True)

    for host in retained:
        host.release_managed_execution("bench.relay_stream_handoff")
    producer.terminate()
    producer.wait(timeout=30)

    total = len(delays_us)
    delays = sorted(delays_us) or [0.0]
    extra = n * (chunks - short)
    marginal_cpu = (p_long["cpu"] - p_short["cpu"]) / max(1, extra)
    return {
        "variant": variant, "n": n, "chunks": total, "expected": n * chunks, "errors": errors[:5],
        "error_count": len(errors), "paths": paths,
        "p50_ms": statistics.median(delays) / 1000,
        "p99_ms": delays[min(len(delays) - 1, int(len(delays) * 0.99))] / 1000,
        "cpu_us": marginal_cpu * 1e6,
        "cs": None if p_long["cs"] is None else (p_long["cs"] - p_short["cs"]) / max(1, extra),
        "load": p_long["cpu"] / max(1e-9, p_long["wall"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ns", default="1,10,50", help="comma-separated concurrent stream counts")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--chunks", type=int, default=200, help="chunks per stream in the long phase")
    parser.add_argument("--gap-ms", type=float, default=20)
    parser.add_argument("--jitter", type=float, default=0.3)
    parser.add_argument("--repeat", type=int, default=1, help="runs per cell; the median run is reported")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--worker", nargs=2, metavar=("VARIANT", "N"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        variant, n = args.worker[0], int(args.worker[1])
        print("RESULT " + json.dumps(run_worker(variant, n, args.chunks, args.gap_ms / 1000, args.jitter, args.seed)))
        return

    print(f"{args.chunks} chunks/stream, gap {args.gap_ms:.0f} ms ±{args.jitter * 100:.0f}%, random start offsets, "
          f"median of {args.repeat}, {sys.platform}, Python {sys.version.split()[0]}")
    for n in [int(x) for x in args.ns.split(",")]:
        print(f"\n{n} concurrent stream(s):")
        print(f"  {'variant':10s} {'p50 ms':>8s} {'p99 ms':>8s} {'CPU us/chunk':>13s} {'ctx sw/chunk':>13s} "
              f"{'CPU load':>9s} {'chunks':>11s} {'errors':>7s}  path managed/async-client")
        for variant in args.variants.split(","):
            runs = []
            for r in range(args.repeat):
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker", variant, str(n), "--chunks", str(args.chunks),
                     "--gap-ms", str(args.gap_ms), "--jitter", str(args.jitter), "--seed", str(args.seed + r)],
                    capture_output=True, text=True, timeout=args.timeout, cwd=REPO)
                line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
                if line is None:
                    print(f"  {variant:10s} FAILED (exit {proc.returncode}): {proc.stderr.strip()[-400:]}")
                    break
                runs.append(json.loads(line[7:]))
            if not runs:
                continue
            r = sorted(runs, key=lambda x: x["p99_ms"])[len(runs) // 2]
            cs = "n/a" if r["cs"] is None else f"{r['cs']:.2f}"
            print(f"  {variant:10s} {r['p50_ms']:8.3f} {r['p99_ms']:8.3f} {r['cpu_us']:13.1f} {cs:>13s} "
                  f"{r['load']:9.2f} {r['chunks']:5d}/{r['expected']:<5d} {r['error_count']:7d}  "
                  f"{r['paths']['managed']}/{r['paths']['async_client']}")
            for err in r["errors"]:
                print(f"      error: {err}")


if __name__ == "__main__":
    main()
