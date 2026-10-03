#!/usr/bin/env python3
"""Remote half of the autoresearch harness: runs on the DGX Spark's mirror of this repo.

Steps, in order (a fresh server start before each probe group, so no probe's KV leaks into the next):
  1. ./start.sh restart  -- reads the synced .env, re-applies NATIVE_CONTEXT to the cached
                            checkpoint config (scripts/config.sh), pulls or rebuilds the image
                            when patches changed, waits for the API, smoke-tests it.
  2. tools/contextprobe.py <target> c1 1 16384  -- C1: one greedy needle at ~CONTEXT - 2048 prompt
                                                   tokens, thinking on, with the visioncheck image in
                                                   the prompt. The reply is the passphrase, a sentence
                                                   naming the image's shapes, then a 500-word essay.
  3. restart, then tools/contextprobe.py <target> c2 2 16384 -- C2: two concurrent needles,
   distinct prompts (no stream's KV is reused by the other), same thinking and image.
  4. tools/visioncheck.py -- thinking on; the vision tower still sees a red circle and a blue square.

stdout: the probes' METRIC lines (c1_*, c2_*, vision_ok, max_context) plus per-concurrency scores
and the primary, computed here:
  METRIC <c>_perf_tps=<geomean of that run's per-stream prefill and decode rates>
  METRIC perf_tps=<geomean of the two legs: a run is as fast as its slowest concurrency>
A failed probe contributes 0, so a failed run scores 0. Thinking and vision are required: a probe
that did not think, did not report the image, or missed the passphrase exits 1, and so does a failed
visioncheck. Exit 3 when the server will not start, 1 when a probe or the vision check failed.

Usage: python3 tools/af_run.py <target_tokens> <port>
"""
import math
import os
import subprocess
import sys


def tail(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return f"(no {path})\n"


def probe(target: int, prefix: str, clients: int, max_reply: int, api_url: str) -> tuple[int, str]:
    """One contextprobe run: (exit code, stdout). Probe stderr goes to /tmp/af_probe.err."""
    env = dict(os.environ, API_URL=api_url)
    with open("/tmp/af_probe.err", "ab") as err:
        proc = subprocess.run(
            [sys.executable, "tools/contextprobe.py", str(target), prefix, str(clients), str(max_reply)],
            stdout=subprocess.PIPE, text=True, stderr=err, env=env)
    return proc.returncode, proc.stdout or ""


def metrics(text: str) -> dict[str, float]:
    vals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("METRIC ") and "=" in line:
            key, _, val = line[7:].partition("=")
            try:
                vals[key] = float(val)
            except ValueError:
                pass
    return vals


def restart() -> None:
    """Fresh server start: clean stream state, so no probe's KV leaks into the next."""
    print("af_run: ./start.sh restart", flush=True)
    with open("/tmp/af_start.log", "wb") as start_log:
        rc = subprocess.run(["./start.sh", "restart"], stdout=start_log, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.stderr.write("af_run: start.sh failed\n" + tail("/tmp/af_start.log"))
        sys.exit(3)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(2)
    target, port = int(sys.argv[1]), sys.argv[2]
    api_url = f"http://127.0.0.1:{port}"

    restart()
    ec1, out1 = probe(target, "c1", 1, 16384, api_url)
    restart()
    ec2, out2 = probe(target, "c2", 2, 16384, api_url)

    vision = subprocess.run([sys.executable, "tools/visioncheck.py"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=dict(os.environ, API_URL=api_url))
    vision_ok = vision.returncode == 0
    sys.stdout.write(vision.stdout or "")
    if vision.stderr:
        sys.stderr.write(vision.stderr)
    print(f"METRIC vision_ok={1 if vision_ok else 0}")

    sys.stdout.write(out1)
    sys.stdout.write(out2)
    if ec1 or ec2:
        sys.stderr.write(tail("/tmp/af_probe.err", 20))

    vals = {**metrics(out1), **metrics(out2)}
    scores = {}
    for c in ("c1", "c2"):
        p, d = vals.get(f"{c}_prefill_tps", 0.0), vals.get(f"{c}_decode_tps", 0.0)
        scores[c] = math.sqrt(p * d) if p > 0 and d > 0 else 0.0
    print(f"METRIC c1_perf_tps={scores['c1']:.1f}")
    print(f"METRIC c2_perf_tps={scores['c2']:.1f}")
    c1, c2 = scores["c1"], scores["c2"]
    print(f"METRIC perf_tps={math.sqrt(c1 * c2) if c1 > 0 and c2 > 0 else 0.0:.1f}")
    if not vision_ok:
        sys.stderr.write("af_run: vision check did not pass with thinking on\n")
    sys.exit(max(ec1, ec2, 0 if vision_ok else 1))


if __name__ == "__main__":
    main()
