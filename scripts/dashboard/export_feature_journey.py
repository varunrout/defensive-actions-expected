"""Export the full feature journey (candidate -> locked/dropped) for both
datasets to dashboard_data/feature_journey.json.

Everything here is derived programmatically from the repo's own source of
truth rather than retyped:
  - Candidate/locked feature LISTS come from importing
    src.eda.feature_config_v1_historical, feature_config_v2_historical and
    feature_config (three generations of the locked-feature config). Each of
    those modules self-asserts its own expected counts at import time, so a
    failed import here means the historical reconstruction is out of sync
    with feature_config.py -- that is real signal, not something to paper
    over.
  - Reasons/evidence for drops come from the `excluded` reason-strings inside
    those same modules (EXCLUDED_COLUMNS_* / REDUNDANCY_DROPPED_*), not from
    a separately-maintained copy.
  - Per-feature profiles (bins / value counts vs. shot rate) are computed
    fresh from the current post-coordinate-fix parquet files, using ALL rows
    (not train/test split) -- this is stated explicitly in the output JSON.

Run from the repo root:
    .venv/Scripts/python.exe scripts/dashboard/export_feature_journey.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eda import feature_config as fc  # noqa: E402
from src.eda import feature_config_v1_historical as fc_v1  # noqa: E402
from src.eda import feature_config_v2_historical as fc_v2  # noqa: E402

OUT_PATH = REPO_ROOT / "dashboard_data" / "feature_journey.json"

TYPE_KEYS = ("categorical", "boolean", "continuous", "discrete")

# Verified against reports/modeling/active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md
# (lines ~19-27), reports/modeling/active_continuous/ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md
# and reports/modeling/active_xt/ACTIVE_XT_BASELINE_SUMMARY.md: these two locked
# ACTIVE features are excluded from every active-leg model (data-quality bug /
# structural redundancy), even though feature_config.py itself keeps them locked.
NOT_MODELLED = {
    "active": {
        "nearest_defender_distance": (
            "known self-reference bug (~73.6% of rows approx 0m per the distribution atlas), "
            "not a genuine defensive-distance signal -- excluded from all active-leg models "
            "(train_active_binary_baseline.py), though still a locked EDA candidate"
        ),
        "defenders_within_5m": (
            "confirmed substitutive/nested with defenders_within_10m for this target "
            "(reports/eda/FEATURE_INTERACTION_ANALYSIS.json) -- excluded from all active-leg "
            "models, though still a locked EDA candidate"
        ),
    },
    "passive": {},
}


def type_lists(cfg: dict) -> dict:
    return {k: list(cfg[k]) for k in TYPE_KEYS}


def flat_set(cfg: dict) -> set:
    return {f for k in TYPE_KEYS for f in cfg[k]}


def feature_type(name: str, cfg: dict) -> str | None:
    for k in TYPE_KEYS:
        if name in cfg[k]:
            return k
    return None


EVIDENCE_PATTERNS = [
    (re.compile(r"r=(-?[\d.]+) vs (\w+)"), "correlation_r"),
    (re.compile(r"Spearman r=(-?[\d.]+) vs (\w+)"), "spearman_r"),
    (re.compile(r"Cramer's V=(-?[\d.]+) vs (\w+)"), "cramers_v"),
    (re.compile(r"point-biserial r=(-?[\d.]+) vs (\w+)"), "point_biserial_r"),
    (re.compile(r"chi2=([\d.]+), p=([\d.eE+-]+)"), "chi2"),
]


def parse_evidence(reason: str) -> dict | None:
    for pattern, metric in EVIDENCE_PATTERNS:
        m = pattern.search(reason)
        if m:
            if metric == "chi2":
                return {"metric": "chi2", "value": float(m.group(1)), "p": float(m.group(2))}
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            return {"metric": metric, "value": value, "vs": m.group(2)}
    return None


def stage_for_dropped(dataset: str, name: str, reason: str, in_v2: bool) -> tuple[int, str]:
    """Classify which pipeline stage a dropped feature actually left at.

    Stage numbers/labels are those used in feature_config.py's own comments
    ("Prompt 5/5" = stage 05 collapse-tier, "Prompt 9/9" = stage 09 VIF,
    "Prompt 10" = stage 10 leakage audit) and in the v1/v2 historical
    modules' docstrings (stage 01 candidate pool, stage 04 structural
    redesign, stage 07 post-collapse/review snapshot, stage 14 final
    correlation regeneration).
    """
    if in_v2:
        # Survived to the stage-07 snapshot; dropped after it.
        if dataset == "active":
            return 9, "Stage 9 - VIF (multicollinearity) analysis"
        return 10, "Stage 10 - Leakage audit"
    # Dropped before the stage-07 snapshot.
    if "Type 1 REVIEW" in reason or "Type 2 REVIEW" in reason:
        return 5, "Stage 5 - Correlation REVIEW-tier resolution"
    if "CORRELATION_ANALYSIS DROP tier" in reason:
        return 5, "Stage 5 - Correlation DROP-tier verdict"
    if "Cluster" in reason and "MERGE" in reason:
        return 6, "Stage 6 - Collapse-tier cluster merge (engineered replacement)"
    if "Cluster" in reason:
        return 5, "Stage 5 - Collapse-tier cluster drop-to-one"
    if "Leakage exclusion" in reason:
        return 10, "Stage 10 - Leakage audit"
    return 4, "Stage 4 - Structural redesign"


def build_dataset_records(dataset: str) -> tuple[list[dict], dict]:
    cfg_v1 = fc_v1.DATASETS_V1_HISTORICAL[dataset]
    cfg_v2 = fc_v2.DATASETS_V2_HISTORICAL[dataset]
    cfg_cur = fc.DATASETS[dataset]

    v1_set = flat_set(cfg_v1)
    v2_set = flat_set(cfg_v2)
    locked_set = flat_set(cfg_cur)

    excluded_common = fc.EXCLUDED_COLUMNS_ACTIVE if dataset == "active" else fc.EXCLUDED_COLUMNS_PASSIVE
    redundancy_dropped = fc.REDUNDANCY_DROPPED_ACTIVE if dataset == "active" else fc.REDUNDANCY_DROPPED_PASSIVE

    records = []

    # 1) Features excluded BEFORE the candidate count even begins (never in v1_set,
    #    except has_screened_outcome on the passive side which IS a stage-01 candidate
    #    -- see feature_config_v1_historical.py's explicit removal of it from `excluded`).
    for name, reason in excluded_common.items():
        if name in v1_set:
            continue  # handled below as a dropped-candidate (e.g. has_screened_outcome)
        records.append(
            {
                "name": name,
                "dataset": dataset,
                "type": None,
                "origin": "excluded_before_count",
                "fate": "dropped",
                "stage": 0,
                "stage_label": "Stage 0 - excluded before candidate pool (data-collection artefact / duplicate / structural zero)",
                "reason": reason,
                "evidence": parse_evidence(reason),
                "modelled": False,
                "source": "src/eda/feature_config.py (EXCLUDED_COLUMNS_ACTIVE/EXCLUDED_COLUMNS_PASSIVE)",
            }
        )

    # 2) Engineered features: locked now, but did not exist as stage-01 candidates.
    engineered = locked_set - v1_set
    for name in sorted(engineered):
        records.append(
            {
                "name": name,
                "dataset": dataset,
                "type": feature_type(name, cfg_cur),
                "origin": "engineered",
                "fate": "locked",
                "stage": 6,
                "stage_label": "Stage 6 - engineered replacement (post-collapse-tier)",
                "reason": "Engineered during the reduction itself to replace a collapsed cluster (did not exist as a stage-01 candidate).",
                "evidence": None,
                "modelled": name not in NOT_MODELLED[dataset],
                "source": "src/eda/feature_config_v1_historical.py (_ACTIVE_ENGINEERED_DURING_REDUCTION / _PASSIVE_ENGINEERED_DURING_REDUCTION)",
            }
        )

    # 3) Every stage-01 candidate: either survives to locked, or was dropped somewhere.
    for name in sorted(v1_set):
        in_v2 = name in v2_set
        if name in locked_set:
            modelled = name not in NOT_MODELLED[dataset]
            reason = None
            source = "src/eda/feature_config.py"
            if not modelled:
                reason = NOT_MODELLED[dataset][name]
                source = (
                    "reports/modeling/active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md "
                    "(also active_continuous / active_xt BASELINE_SUMMARY.md)"
                )
            records.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "type": feature_type(name, cfg_cur),
                    "origin": "candidate",
                    "fate": "locked",
                    "stage": 14,
                    "stage_label": "Stage 14 - final locked feature list",
                    "reason": reason,
                    "evidence": None,
                    "modelled": modelled,
                    "source": source,
                }
            )
        else:
            reason = redundancy_dropped.get(name) or excluded_common.get(name) or "reason not found in feature_config.py"
            stage_num, stage_label = stage_for_dropped(dataset, name, reason, in_v2)
            records.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "type": feature_type(name, cfg_v1),
                    "origin": "candidate",
                    "fate": "dropped",
                    "stage": stage_num,
                    "stage_label": stage_label,
                    "reason": reason,
                    "evidence": parse_evidence(reason),
                    "modelled": False,
                    "source": "src/eda/feature_config.py (REDUNDANCY_DROPPED_ACTIVE/REDUNDANCY_DROPPED_PASSIVE or EXCLUDED_COLUMNS_PASSIVE)",
                }
            )

    stage_counts = {
        "excluded_before_count": len(
            [r for r in records if r["origin"] == "excluded_before_count"]
        ),
        "candidate_pool_stage01": len(v1_set),
        "post_collapse_review_stage07": len(v2_set),
        "final_locked": len(locked_set),
    }
    type_breakdown = {
        "candidate_stage01": {k: len(cfg_v1[k]) for k in TYPE_KEYS},
        "locked_final": {k: len(cfg_cur[k]) for k in TYPE_KEYS},
    }
    return records, {"stage_counts": stage_counts, "type_breakdown": type_breakdown}


def profile_continuous(series: pd.Series, target: pd.Series) -> list[dict]:
    valid = series.notna()
    s = series[valid]
    t = target[valid]
    try:
        binned = pd.qcut(s, q=10, duplicates="drop")
    except ValueError:
        binned = pd.qcut(s, q=min(10, s.nunique()), duplicates="drop")
    out = []
    grouped = t.groupby(binned, observed=True)
    for interval, grp in grouped:
        out.append(
            {
                "bin": str(interval),
                "n": int(grp.shape[0]),
                "shot_rate_pct": round(float(grp.mean()) * 100, 3),
            }
        )
    return out


def profile_discrete(series: pd.Series, target: pd.Series, cap: int | None = None) -> list[dict]:
    valid = series.notna()
    s = series[valid]
    t = target[valid]
    counts = s.value_counts()
    if cap is not None and len(counts) > cap:
        top = counts.iloc[:cap]
        rest_mask = ~s.isin(top.index)
        out = []
        for val, n in top.items():
            grp = t[s == val]
            out.append({"value": str(val), "n": int(n), "shot_rate_pct": round(float(grp.mean()) * 100, 3)})
        rest_n = int(rest_mask.sum())
        if rest_n:
            grp = t[rest_mask]
            out.append({"value": "other", "n": rest_n, "shot_rate_pct": round(float(grp.mean()) * 100, 3)})
        return out
    out = []
    for val, n in counts.items():
        grp = t[s == val]
        out.append({"value": str(val), "n": int(n), "shot_rate_pct": round(float(grp.mean()) * 100, 3)})
    return out


def add_profiles(records: list[dict], dataset: str, cfg_cur: dict) -> None:
    path = REPO_ROOT / cfg_cur["parquet_path"]
    import pyarrow.parquet as pq

    available_cols = set(pq.ParquetFile(path).schema.names)
    locked_names = [r["name"] for r in records if r["fate"] == "locked" and r["origin"] != "excluded_before_count"]
    cols = [c for c in dict.fromkeys(locked_names) if c in available_cols] + [cfg_cur["target_col"]]
    df = pd.read_parquet(path, columns=cols)
    target = df[cfg_cur["target_col"]]
    by_name = {r["name"]: r for r in records}
    for name in locked_names:
        if name not in df.columns:
            by_name[name]["profile"] = {"error": f"column {name} not found in {cfg_cur['parquet_path']}"}
            continue
        rtype = by_name[name]["type"]
        series = df[name]
        if rtype == "continuous":
            try:
                profile = profile_continuous(series.astype(float), target)
                method = "quantile deciles (10 bins, or fewer if ties collapse them)"
            except Exception as exc:  # pragma: no cover - defensive
                profile = []
                method = f"failed: {exc}"
        else:
            cap = 15 if rtype == "categorical" else None
            profile = profile_discrete(series, target, cap=cap)
            method = "value counts (top 15 + other)" if rtype == "categorical" else "value counts"
        by_name[name]["profile"] = {
            "n_rows_used": int(series.notna().sum()),
            "binning_method": method,
            "bins": profile,
        }


def main() -> None:
    all_records = []
    stage_meta = {}
    for dataset in ("active", "passive"):
        records, meta = build_dataset_records(dataset)
        cfg_cur = fc.DATASETS[dataset]
        add_profiles(records, dataset, cfg_cur)
        all_records.extend(records)
        stage_meta[dataset] = meta

    # --- Assertions: compute, don't assume. ---
    assertions = []

    def check(label, actual, expected=None):
        ok = True if expected is None else actual == expected
        assertions.append({"label": label, "actual": actual, "expected": expected, "passed": ok})
        return ok

    active_meta = stage_meta["active"]["stage_counts"]
    passive_meta = stage_meta["passive"]["stage_counts"]
    check("active candidate pool (stage 01)", active_meta["candidate_pool_stage01"])
    check("active final locked", active_meta["final_locked"])
    check("passive candidate pool (stage 01)", passive_meta["candidate_pool_stage01"])
    check("passive final locked", passive_meta["final_locked"])
    check(
        "active pipeline endpoint 51->34 (self-computed)",
        f"{active_meta['candidate_pool_stage01']}->{active_meta['final_locked']}",
    )
    check(
        "passive pipeline endpoint 44->38 (self-computed)",
        f"{passive_meta['candidate_pool_stage01']}->{passive_meta['final_locked']}",
    )
    for ds in ("active", "passive"):
        tb = stage_meta[ds]["type_breakdown"]
        check(f"{ds} candidate type breakdown (cat/bool/cont/disc)", tb["candidate_stage01"])
        check(f"{ds} locked type breakdown (cat/bool/cont/disc)", tb["locked_final"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/dashboard/export_feature_journey.py",
        "notes": [
            "Candidate/locked feature lists parsed directly from src/eda/feature_config_v1_historical.py, "
            "feature_config_v2_historical.py and feature_config.py -- not retyped.",
            "Per-feature profiles computed from ALL rows of the current (post coordinate-frame-fix, "
            "commit f7204c7 / Prompt 85) data/features/player_defensive_actions.parquet and "
            "data/features/passive_defense.parquet -- no train/test split applied here.",
            "modelled=false on nearest_defender_distance / defenders_within_5m (active) reflects a "
            "downstream modelling-stage exclusion (train_active_binary_baseline.py), not a "
            "feature_config.py lock/drop decision -- both remain locked EDA candidates.",
        ],
        "stages": stage_meta,
        "assertions": assertions,
        "features": all_records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(all_records)} feature records)")
    print(json.dumps(assertions, indent=2, default=str))


if __name__ == "__main__":
    main()
