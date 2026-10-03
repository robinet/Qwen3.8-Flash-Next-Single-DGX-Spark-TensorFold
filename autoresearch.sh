#!/bin/sh
# Autoresearch harness entrypoint. The logic is Python, not sh:
#   ../.local-tools/autoresearch.py  syncs this tree to the DGX Spark and starts the remote run
#   tools/af_run.py                  (on the Spark) restarts the server and benchmarks C1 and C2
#   tools/contextprobe.py            (on the Spark) the greedy-needle probes that print the METRICs
# Primary metric: perf_tps = max over C1, C2 of geomean(per-stream prefill, per-stream decode).
# stdout carries the METRIC lines; exit 0 only when both context probes completed.
# Everything this script starts is a Windows native (python3.exe, uv.exe, ssh.exe via the recipe's
# .local-tools); no WSL, no other Linux userland. AR_TOOLS overrides where .local-tools lives
# (default: this tree's parent). Strict POSIX sh: this host's `sh` is a native POSIX layer, and
# `bash` may not exist at all.
set -eu
case "$0" in
  *\\*) cd "${0%\\*}" ;;
  */*)  cd "${0%/*}" ;;
esac
ROOT=$(pwd -P)
case "$ROOT" in
  *\\*) PARENT=${ROOT%\\*} ;;
  */*)  PARENT=${ROOT%/*} ;;
  *)    PARENT=. ;;
esac
TOOLS=${AR_TOOLS:-$PARENT/.local-tools}
[ -f "$TOOLS/autoresearch.py" ] || { printf 'autoresearch.sh: %s/autoresearch.py not found\n' "$TOOLS" >&2; exit 2; }
# The Windows interpreter takes whatever dialect pwd reported (C:\..., /c/..., /mnt/c/...);
# .local-tools/autoresearch.py normalises it.
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-project "$TOOLS/autoresearch.py" "$ROOT"
fi
exec python3 "$TOOLS/autoresearch.py" "$ROOT"
