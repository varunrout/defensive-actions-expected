"""Fit passive-defense archetype clustering pipelines, persist them, and
write defender_archetype_cluster_id/name labels back across the full
passive_defense.parquet population."""
from __future__ import annotations

from dax.analysis.passive_archetypes import main


if __name__ == "__main__":
    raise SystemExit(main())
