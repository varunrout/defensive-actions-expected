"""Deprecated compatibility wrapper for the canonical pipeline entry point."""

from __future__ import annotations

import sys

from dax.pipeline.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["--stage", "prepare-data", *sys.argv[1:]]))
