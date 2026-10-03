#!/usr/bin/env python3
"""Prune the Spark mirror's patches/ to the recipe's manifest.

The sync writes .af-patch-list (one relative patch path per line) from the working tree before
tarring it; a patch this repo no longer keeps must not survive in the mirror or it would be baked
into a rebuilt image. Run on the mirror after the tar lands:

    python3 tools/prune_patches.py
"""
import pathlib

KEEP = pathlib.Path(".af-patch-list")


def main() -> None:
    keep = set(KEEP.read_text(encoding="utf-8").split()) if KEEP.exists() else set()
    for path in sorted(pathlib.Path("patches").rglob("*.patch")):
        if str(path) not in keep:
            path.unlink()
            print(f"pruned stale patch: {path}")


if __name__ == "__main__":
    main()
