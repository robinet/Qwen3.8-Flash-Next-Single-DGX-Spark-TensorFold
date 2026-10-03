#!/usr/bin/env python3
"""Long-context probe for the autoresearch harness: greedy needles in ~N tokens of synthetic prose, on
one or more concurrent streams.

Usage: tools/contextprobe.py [target_tokens] [prefix] [clients] [max_reply]
  target_tokens: aim for this many prompt tokens per stream. Default: CONTEXT from the .env next to
                 start.sh minus 2,048 (room for the reply), so the probe exercises the configured window.
  prefix:  metric and label prefix (e.g. c1 or c2); METRIC names are <prefix>_prefill_tps and friends.
  clients: streams started on a barrier. Each stream gets its own prompt (seed 777 + index), so no
           stream's KV can be reused by another. Default 1.
  max_reply: per-request max_tokens. Default 2048.
  API_URL: the server (default http://127.0.0.1:<PORT from the .env next to start.sh>).

Prints METRIC lines (all 0 on failure). The *_tps rates are per stream (with one client, the whole
stream is one stream); agg is the all-streams e2e rate, the C2 table's "aggregate":
  METRIC max_context=<tokens>     prompt tokens actually served (the first stream)
  METRIC <prefix>_prefill_tps=    prompt tokens per stream / wall prefill time (last stream to finish)
  METRIC <prefix>_decode_tps=     reply tokens per stream / wall decode time
  METRIC <prefix>_agg_tps=        all prompt and reply tokens / total wall time
  METRIC <prefix>_ttft_s=         the last stream's time to first token
  METRIC <prefix>_needle=         1: every stream answered the hidden passphrase
  METRIC <prefix>_mtp_yield=      1 + accepted drafts / rounds across the streams
Exit code 0 when every request completed, 1 otherwise.

The reply is the passphrase line plus a 500-word essay, so the decode is long enough to measure a
rate against the full window (a greedy needle reply otherwise ends after ~10 tokens).

Thinking is on (the server default, and the request asks for it): the reasoning is in reasoning_content
and counts in the decode. Every prompt also carries the visioncheck image, and the answer has to name its
shapes before the essay, so the measured prefill includes the vision tower. A stream that did not think, did not report
the image, or missed the passphrase fails the probe.
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing bench


def _env_line(name: str) -> str | None:
    """One KEY=VALUE from the recipe's .env (next to start.sh); commented lines skipped."""
    env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].split("#", 1)[0].strip()
    except OSError:
        return None


# bench.py builds API_URL from PORT at import time; the recipe's .env wins over its 8888 default.
if not os.environ.get("API_URL"):
    _port = _env_line("PORT")
    if _port:
        os.environ["PORT"] = _port
from bench import API_URL, URL, prose  # noqa: E402
from visioncheck import png  # noqa: E402

SECRET = "violet-harbor-7291"
# The same drawn image visioncheck sends: a red circle and a blue square. One encoding, shared by every
# stream, so the vision tokens are identical and the request stays deterministic.
IMAGE_URL = "data:image/png;base64," + base64.b64encode(png()).decode()
VISION_WORDS = ("red", "circle", "blue", "square")
TOKENS_PER_SIZE = 0.79                   # bench.prose(N) comes out at ~0.79 x N tokens
TIMEOUT = 1800                           # a 512k-token prefill takes ~6 minutes; two of them at once, more


def context_from_env() -> int:
    """CONTEXT from the recipe's .env (next to start.sh)."""
    value = _env_line("CONTEXT")
    return int(value) if value else 262144


def served_name() -> str:
    req = urllib.request.Request(API_URL + "/v1/models", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["data"][0]["id"]


def metric_names(prefix: str) -> list[str]:
    return ["max_context", f"{prefix}_prefill_tps", f"{prefix}_decode_tps", f"{prefix}_agg_tps",
            f"{prefix}_ttft_s", f"{prefix}_needle", f"{prefix}_thinking", f"{prefix}_vision",
            f"{prefix}_mtp_yield"]


def prompt_for(target: int, client: int) -> str:
    size = int(target / TOKENS_PER_SIZE)
    hay = prose(size, 777 + client).split(". ")
    at = int(len(hay) * 0.6)
    hay.insert(at, f"Remember this: the secret passphrase is {SECRET}")
    return (". ".join(hay)
            + "\n\nWhat is the secret passphrase mentioned in the text above? First reply with the passphrase "
              "only on its own line, then one sentence naming the shapes in the attached image and the colour "
              "of each, then write a 500-word essay titled 'Notes on the text above'.")


def one(model: str, prompt: str, max_reply: int) -> dict:
    """One non-streamed greedy request; HTTP, SSE and timeout errors are failures, not a substituted time."""
    body = {"model": model, "max_tokens": max_reply, "temperature": 0, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "text", "text": prompt}]}]}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            reply = json.load(resp)
    except Exception as exc:
        return {"ok": False, "error": str(exc) or type(exc).__name__}
    tf = reply.get("tensorfold") or {}
    usage = reply.get("usage") or {}
    message = (reply.get("choices") or [{}])[0].get("message", {}) or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    decode_s = float(tf.get("decode_s") or 0.0)
    wall = time.time() - t0
    prefill_s = float(tf.get("prefill_s") or (wall - decode_s if wall > decode_s else wall))
    return {"ok": True, "wall": wall,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "prefill_s": prefill_s, "decode_s": decode_s,
            "drafted": int(tf.get("drafted") or 0), "accepted": int(tf.get("accepted") or 0),
            "rounds": int(tf.get("rounds") or 0),
            "needle": SECRET in content,
            "thinking": reasoning_tokens > 0 or bool(reasoning.strip()),
            "vision": all(word in content.lower() for word in VISION_WORDS),
            "reasoning_tokens": reasoning_tokens,
            "tail": content.strip()[-80:]}


def run(target: int, prefix: str, clients: int, max_reply: int) -> None:
    model = served_name()
    prompts = [prompt_for(target, i) for i in range(clients)]
    barrier, out = threading.Barrier(clients), [None] * clients

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            out[i] = {"ok": False, "error": "start barrier broken"}
            return
        out[i] = one(model, prompts[i], max_reply)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bad = [r for r in out if not r.get("ok")]
    if bad:
        print(f"{prefix}: {len(bad)}/{len(out)} failed: "
              + "; ".join(str(r.get("error")) for r in bad[:3]), file=sys.stderr)
        for name in metric_names(prefix):
            print(f"METRIC {name}=0")
        sys.exit(1)

    ptot = sum(r["prompt_tokens"] for r in out)
    ctot = sum(r["completion_tokens"] for r in out)
    prefill_wall = max(r["prefill_s"] for r in out)
    decode_wall = max(r["decode_s"] for r in out)
    ttft = max(r["prefill_s"] for r in out)           # first token = prefill done
    drafted = sum(r["drafted"] for r in out)
    accepted = sum(r["accepted"] for r in out)
    rounds = sum(r["rounds"] for r in out)
    yield_ = 1.0 + (accepted / rounds if rounds else (accepted / drafted if drafted else 0.0))
    needle = all(r["needle"] for r in out)
    thinking = all(r["thinking"] for r in out)
    vision = all(r["vision"] for r in out)

    per = float(len(out))
    print(f"METRIC max_context={out[0]['prompt_tokens']}")
    print(f"METRIC {prefix}_prefill_tps={(ptot / per) / prefill_wall:.1f}" if prefill_wall else
          f"METRIC {prefix}_prefill_tps=0")
    print(f"METRIC {prefix}_decode_tps={(ctot / per) / decode_wall:.1f}" if decode_wall else
          f"METRIC {prefix}_decode_tps=0")
    total_wall = prefill_wall + decode_wall
    print(f"METRIC {prefix}_agg_tps={(ptot + ctot) / total_wall:.1f}" if total_wall else
          f"METRIC {prefix}_agg_tps=0")
    print(f"METRIC {prefix}_ttft_s={ttft:.2f}")
    print(f"METRIC {prefix}_needle={1 if needle else 0}")
    print(f"METRIC {prefix}_thinking={1 if thinking else 0}")
    print(f"METRIC {prefix}_vision={1 if vision else 0}")
    print(f"METRIC {prefix}_mtp_yield={yield_:.3f}")
    for i, r in enumerate(out):
        print(f"{prefix}: stream {i}: {r['prompt_tokens']} tok prompt, prefill {r['prefill_s']:.1f} s, "
              f"{r['completion_tokens']} reply tok in {r['decode_s']:.1f} s, total {r['wall']:.1f} s, "
              f"needle {'CORRECT' if r['needle'] else 'WRONG'}, "
              f"thinking {'ON' if r['thinking'] else 'OFF'} ({r['reasoning_tokens']} reasoning tok), "
              f"vision {'OK' if r['vision'] else 'FAIL'} ({r['tail']!r})", flush=True)
    if not (needle and thinking and vision):
        print(f"{prefix}: quality failed: needle={int(needle)} thinking={int(thinking)} "
              f"vision={int(vision)}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else max(4096, context_from_env() - 2048)
    prefix = sys.argv[2] if len(sys.argv) > 2 else "probe"
    clients = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_reply = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    if clients < 1 or max_reply < 1:
        print(f"{prefix}: clients and max_reply must be >= 1", file=sys.stderr)
        for name in metric_names(prefix):
            print(f"METRIC {name}=0")
        sys.exit(1)
    try:
        run(target, prefix, clients, max_reply)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"{prefix}: probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        for name in metric_names(prefix):
            print(f"METRIC {name}=0")
        sys.exit(1)


if __name__ == "__main__":
    main()
