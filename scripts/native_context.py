#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pin the cached checkpoint's context limits to NATIVE_CONTEXT.

TensorFold reads the prompt/reply window from ``config.json`` (``max_position_embeddings``, top
level and ``text_config``). Since v0.6.2 the server also runs prompts through the Hugging Face
tokenizer, which warns or refuses past the ``model_max_length`` pinned in
``tokenizer_config.json``. Both files are rewritten in place, idempotently, at every start.

Usage (run by scripts/config.sh)::

    uv run --script scripts/native_context.py \
        --cache-dir ~/.cache/huggingface \
        --model-id Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
        --native 524288
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rewrite_snapshot(snap: Path, native: int) -> list[str]:
    """Rewrite the context limits in one snapshot; returns the files it changed."""
    changed: list[str] = []
    cfg_path = snap / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if not (cfg.get("max_position_embeddings") == native
                and (cfg.get("text_config") or {}).get("max_position_embeddings") == native):
            cfg["max_position_embeddings"] = native
            if isinstance(cfg.get("text_config"), dict):
                cfg["text_config"]["max_position_embeddings"] = native
            cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
            changed.append(f"{snap.name}/config.json")
    tc_path = snap / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if tc.get("model_max_length") != native:
            tc["model_max_length"] = native
            tc_path.write_text(json.dumps(tc, indent=2) + "\n")
            changed.append(f"{snap.name}/tokenizer_config.json")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True,
                        help="Hugging Face cache root (its hub/ holds the model)")
    parser.add_argument("--model-id", required=True,
                        help="checkpoint id, e.g. Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    parser.add_argument("--native", type=int, required=True,
                        help="token window to pin both files to")
    args = parser.parse_args()

    root = Path(args.cache_dir) / "hub" / ("models--" + args.model_id.replace("/", "--"))
    snaps = sorted({p.parent for p in root.glob("snapshots/*/config.json")})
    changed: list[str] = []
    for snap in snaps:
        changed.extend(rewrite_snapshot(snap, args.native))
    if changed:
        print(f"[config] NATIVE_CONTEXT={args.native}: now in place in {', '.join(changed)}")
    elif snaps:
        print(f"[config] NATIVE_CONTEXT={args.native}: already in place")
    else:
        print(f"[config] NATIVE_CONTEXT={args.native}: no snapshot of {args.model_id} under {root}; "
              "the native window is unchanged", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
