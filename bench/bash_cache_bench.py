"""Time pm.shell.bash() and LocalEnvironment commands for whichever pm/shell.py is checked out.

Run once per variant in a fresh process; prints one JSON line.
"""
import json
import os
import statistics
import sys
import tempfile
import time

N = int(os.environ.get("BENCH_N", "20"))
os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-bench-"))
sys.path.insert(0, os.getcwd())

import pm.shell as shell  # noqa: E402

probes = []
_real_probe = shell._bash_starts


def _counting_probe(candidate):
    probes.append(candidate)
    return _real_probe(candidate)


shell._bash_starts = _counting_probe


def ms(seconds):
    return round(seconds * 1000, 3)


t = time.perf_counter()
resolved = shell.bash()
first_call = time.perf_counter() - t
first_probes = len(probes)

calls = []
for _ in range(N):
    t = time.perf_counter()
    shell.bash()
    calls.append(time.perf_counter() - t)
repeat_probes = len(probes) - first_probes

from tools.environments.local import LocalEnvironment  # noqa: E402

t = time.perf_counter()
env = LocalEnvironment(cwd=os.getcwd(), timeout=60)
init_session = time.perf_counter() - t

probes.clear()
commands = []
for _ in range(N):
    t = time.perf_counter()
    result = env.execute("echo ok")
    commands.append(time.perf_counter() - t)
    if result.get("returncode") != 0 or "ok" not in result.get("output", ""):
        raise SystemExit(f"echo failed: {result!r}")
command_probes = len(probes)
env.cleanup()

print(json.dumps({
    "variant": os.environ.get("BENCH_VARIANT", "?"),
    "resolved_bash": resolved,
    "bash_first_call_ms": ms(first_call),
    "bash_repeat_call_median_ms": ms(statistics.median(calls)),
    "bash_probes_first_call": first_probes,
    "bash_probes_next_calls": repeat_probes,
    "local_env_init_ms": ms(init_session),
    "echo_median_ms": ms(statistics.median(commands)),
    "echo_mean_ms": ms(statistics.mean(commands)),
    "bash_probes_during_echo": command_probes,
    "n": N,
}))
