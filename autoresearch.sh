#!/usr/bin/env bash
# Autoresearch harness entrypoint. The logic is Python, not bash:
#   .local-tools/autoresearch.py  syncs this tree to the DGX Spark and starts the remote run
#   tools/af_run.py               (on the Spark) restarts the server and benchmarks C1 and C2
#   tools/contextprobe.py         (on the Spark) the greedy-needle probes that print the METRICs
# Primary metric: perf_tps = max over C1, C2 of geomean(per-stream prefill, per-stream decode).
# stdout carries the METRIC lines; exit 0 only when both context probes completed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-project .local-tools/autoresearch.py
fi
exec python3 .local-tools/autoresearch.py
