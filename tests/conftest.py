"""Pytest configuration for headless chart generation."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
