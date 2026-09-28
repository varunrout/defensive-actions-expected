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
  - Stage numbers/labels for every dropped or engineered candidate are looked
    up from hardcoded per-stage name sets below, verified feature-by-feature
    against reports/analysis/shot_target/EDA_PIPELINE_LOG.html (the
    authoritative stage-numbering source -- stages 04, 05, 06, 09, 10 are the
    only stages that ever change the candidate count) rather than inferred
    from the wording of each reason string, which does not reliably say which
    stage a feature actually left at.
  - Per-feature profiles (bins / value counts vs. shot rate) are computed
    fresh from the current post-coordinate-fix parquet files, using ALL rows
    (not train/test split) -- this is stated explicitly in the output JSON --
    for every feature, not only the 72 locked ones: also the 37 dropped
    candidates, and the 27 pre-count exclusions where the column exists.

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

# ---------------------------------------------------------------------------
# Stage classification, hardcoded from reports/analysis/shot_target/
# EDA_PIPELINE_LOG.html's own per-stage deltas (the authoritative source --
# see prompt 90a). Stages 04, 05, 06, 09 and 10 are the only stages that ever
# change the candidate count; every dropped/engineered candidate is assigned
# to exactly one of them below. Verified: active dropped = 7+10+2 = 19,
# passive dropped = 6+5+6+1 = 18, total 37 -- matches
# len(v1_set - locked_set) for both datasets exactly.
# ---------------------------------------------------------------------------

STAGE_04_ACTIVE_DROPPED = {
    "is_central_lane", "is_wide_lane", "is_high_zone", "is_deep_zone",
    "position_group", "action_family", "distance_to_center_line",
}
STAGE_05_ACTIVE_DROPPED = {
    "distance_to_attacking_goal", "distance_to_defending_goal", "distance_to_defending_box",
    "action_zone", "attacker_centroid_x", "attacker_centroid_y",
    "defender_centroid_x", "defender_centroid_y",
    "events_elapsed_in_possession", "phase_transitions_observed_so_far",
}
STAGE_09_ACTIVE_DROPPED = {"local_numerical_balance_5m", "local_numerical_balance_10m"}

STAGE_04_PASSIVE_DROPPED = {
    "is_central_lane", "screens_top_option", "is_high_zone", "is_deep_zone",
    "zone_defensive_value", "distance_to_center_line",
}
STAGE_05_PASSIVE_DROPPED = {
    "distance_to_attacking_box", "distance_to_attacking_goal",
    "distance_to_defending_box", "distance_to_defending_goal", "defender_zone",
}
STAGE_06_PASSIVE_DROPPED = {
    "top_option_1_target_x", "top_option_1_target_y",
    "top_option_2_target_x", "top_option_2_target_y",
    "top_option_3_target_x", "top_option_3_target_y",
}
STAGE_10_PASSIVE_DROPPED = {"has_screened_outcome"}

STAGE_LABELS = {
    4: "Stage 04 - Structural redesign",
    5: "Stage 05 - Collapse-tier resolution",
    6: "Stage 06 - Drop raw option coordinates",
    9: "Stage 09 - Multicollinearity (VIF)",
    10: "Stage 10 - Leakage audit",
}

# Engineered-during-reduction features: which stage they were introduced at.
STAGE_ENGINEERED = {
    "active": (
        5,
        "Stage 05 - Collapse-tier resolution (Cluster 2 MERGE: attacker/defender centroids "
        "replaced by defender_attacker_gap_x/y)",
    ),
    "passive": (
        4,
        "Stage 04 - Structural redesign (raw option coordinates replaced by ball-relative "
        "top_option_n_{dx,dy,distance_from_ball,angle_from_ball})",
    ),
}


def stage_for_dropped(dataset: str, name: str) -> tuple[int, str]:
    """Look up which pipeline stage a dropped candidate actually left at.

    Deterministic name-based lookup against the hardcoded per-stage sets
    above (themselves verified against EDA_PIPELINE_LOG.html), not inferred
    from the wording of the drop reason string.
    """
    if dataset == "active":
        if name in STAGE_04_ACTIVE_DROPPED:
            return 4, STAGE_LABELS[4] + " (DROP-tier / Type 1&2 REVIEW resolution applied)"
        if name in STAGE_05_ACTIVE_DROPPED:
            return 5, STAGE_LABELS[5] + " (clusters 1-3, drop-to-one)"
        if name in STAGE_09_ACTIVE_DROPPED:
            return 9, STAGE_LABELS[9]
        raise AssertionError(f"active dropped candidate {name!r} not classified into any stage")
    if name in STAGE_04_PASSIVE_DROPPED:
        return 4, STAGE_LABELS[4] + " (DROP-tier resolution; option-coordinate cleanup counted separately at stage 06)"
    if name in STAGE_05_PASSIVE_DROPPED:
        return 5, STAGE_LABELS[5] + " (Cluster 4, drop-to-one)"
    if name in STAGE_06_PASSIVE_DROPPED:
        return 6, STAGE_LABELS[6] + " (ball-relative replacements from stage 04 confirmed working)"
    if name in STAGE_10_PASSIVE_DROPPED:
        return 10, STAGE_LABELS[10]
    raise AssertionError(f"passive dropped candidate {name!r} not classified into any stage")


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


def type_lists(cfg: dict) -> dict:
    return {k: list(cfg[k]) for k in TYPE_KEYS}


def flat_set(cfg: dict) -> set:
    return {f for k in TYPE_KEYS for f in cfg[k]}


def feature_type(name: str, cfg: dict) -> str | None:
    for k in TYPE_KEYS:
        if name in cfg[k]:
            return k
    return None


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
                "stage": None,
                "stage_label": "Before candidate pool - excluded before stage 01 (data-collection artefact / duplicate / structural zero)",
                "reason": reason,
                "evidence": parse_evidence(reason),
                "modelled": False,
                "source": "src/eda/feature_config.py (EXCLUDED_COLUMNS_ACTIVE/EXCLUDED_COLUMNS_PASSIVE)",
            }
        )

    # 2) Engineered features: locked now, but did not exist as stage-01 candidates.
    engineered = locked_set - v1_set
    eng_stage, eng_label = STAGE_ENGINEERED[dataset]
    for name in sorted(engineered):
        records.append(
            {
                "name": name,
                "dataset": dataset,
                "type": feature_type(name, cfg_cur),
                "origin": "engineered",
                "fate": "locked",
                "stage": eng_stage,
                "stage_label": eng_label,
                "reason": "Engineered during the reduction itself to replace a collapsed cluster (did not exist as a stage-01 candidate).",
                "evidence": None,
                "modelled": name not in NOT_MODELLED[dataset],
                "source": "src/eda/feature_config_v1_historical.py (_ACTIVE_ENGINEERED_DURING_REDUCTION / _PASSIVE_ENGINEERED_DURING_REDUCTION)",
            }
        )

    # 3) Every stage-01 candidate: either survives to locked, or was dropped somewhere.
    for name in sorted(v1_set):
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
            stage_num, stage_label = stage_for_dropped(dataset, name)
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

    # --- Per-stage ledger: {stage, label, count_before, dropped, engineered, count_after, source} ---
    ledger = []
    running = len(v1_set)
    # Every stage that CAN change a count (04, 05, 06, 09, 10) is listed for both
    # datasets, even at stages where this particular dataset has no change --
    # so the ledger reproduces the full 51->44->36->36->34->34 /
    # 44->50->45->39->39->38 sequences, not just the stages that moved.
    if dataset == "active":
        # defender_attacker_gap_x/y engineered at stage 05 (Cluster 2 MERGE).
        stage_plan = [
            (4, STAGE_04_ACTIVE_DROPPED, set()),
            (5, STAGE_05_ACTIVE_DROPPED, set(engineered)),
            (6, set(), set()),
            (9, STAGE_09_ACTIVE_DROPPED, set()),
            (10, set(), set()),
        ]
    else:
        # The 12 ball-relative top_option_n_* columns are engineered at stage 04,
        # alongside that stage's drops.
        stage_plan = [
            (4, STAGE_04_PASSIVE_DROPPED, set(engineered)),
            (5, STAGE_05_PASSIVE_DROPPED, set()),
            (6, STAGE_06_PASSIVE_DROPPED, set()),
            (9, set(), set()),
            (10, STAGE_10_PASSIVE_DROPPED, set()),
        ]

    for stage_num, dropped_set, engineered_here in stage_plan:
        count_before = running
        count_after = count_before - len(dropped_set) + len(engineered_here)
        ledger.append(
            {
                "stage": stage_num,
                "label": STAGE_LABELS[stage_num],
                "count_before": count_before,
                "dropped": sorted(dropped_set),
                "engineered": sorted(engineered_here),
                "count_after": count_after,
                "source": "reports/analysis/shot_target/EDA_PIPELINE_LOG.html",
            }
        )
        running = count_after

    return records, {
        "stage_counts": stage_counts,
        "type_breakdown": type_breakdown,
        "ledger": ledger,
    }


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
    """Compute profiles for EVERY feature record where the column exists in
    the current parquet -- locked, dropped-candidate, and (where the column
    still exists) pre-count-excluded alike. Records whose column is not
    present get profile=None with an explicit profile_reason.
    """
    path = REPO_ROOT / cfg_cur["parquet_path"]
    import pyarrow.parquet as pq

    available_cols = set(pq.ParquetFile(path).schema.names)
    all_names = [r["name"] for r in records]
    cols = [c for c in dict.fromkeys(all_names) if c in available_cols] + [cfg_cur["target_col"]]
    df = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
    target = df[cfg_cur["target_col"]]

    for r in records:
        name = r["name"]
        if name not in available_cols:
            r["profile"] = None
            r["profile_reason"] = f"column {name!r} not present in {cfg_cur['parquet_path']}"
            continue
        rtype = r["type"]
        series = df[name]
        try:
            if rtype == "continuous":
                profile = profile_continuous(series.astype(float), target)
                method = "quantile deciles (10 bins, or fewer if ties collapse them)"
            elif rtype == "categorical":
                profile = profile_discrete(series, target, cap=15)
                method = "value counts (top 15 + other)"
            elif rtype in ("boolean", "discrete"):
                profile = profile_discrete(series, target, cap=None)
                method = "value counts"
            else:
                # type is None (e.g. a pre-count exclusion never typed in any
                # feature_config generation) -- fall back to a generic value-count
                # profile rather than skipping it, since the column data exists.
                if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
                    profile = profile_continuous(series.astype(float), target)
                    method = "quantile deciles (10 bins, or fewer if ties collapse them) -- type unresolved, treated as continuous by cardinality"
                else:
                    profile = profile_discrete(series, target, cap=15)
                    method = "value counts (top 15 + other) -- type unresolved, treated as discrete/categorical by cardinality"
        except Exception as exc:  # pragma: no cover - defensive
            r["profile"] = None
            r["profile_reason"] = f"profiling failed: {exc}"
            continue
        r["profile"] = {
            "n_rows_used": int(series.notna().sum()),
            "binning_method": method,
            "bins": profile,
        }
        r["profile_reason"] = None


def main() -> None:
    all_records = []
    stage_meta = {}
    for dataset in ("active", "passive"):
        records, meta = build_dataset_records(dataset)
        cfg_cur = fc.DATASETS[dataset]
        add_profiles(records, dataset, cfg_cur)
        all_records.extend(records)
        stage_meta[dataset] = meta

    # --- Assertions: hardcoded expected values as a verification contract. ---
    # These are checks, not data -- every one of them raises on mismatch.
    assertions = []

    def check(label, actual, expected):
        ok = actual == expected
        assertions.append({"label": label, "actual": actual, "expected": expected, "passed": ok})
        if not ok:
            raise AssertionError(f"Assertion failed: {label}: actual={actual!r} expected={expected!r}")
        return ok

    active_meta = stage_meta["active"]
    passive_meta = stage_meta["passive"]
    active_counts = active_meta["stage_counts"]
    passive_counts = passive_meta["stage_counts"]

    # Endpoints: 51, 34, 44, 38.
    check("active candidate pool (stage 01)", active_counts["candidate_pool_stage01"], 51)
    check("active final locked", active_counts["final_locked"], 34)
    check("passive candidate pool (stage 01)", passive_counts["candidate_pool_stage01"], 44)
    check("passive final locked", passive_counts["final_locked"], 38)
    check(
        "active pipeline endpoint 51->34",
        f"{active_counts['candidate_pool_stage01']}->{active_counts['final_locked']}",
        "51->34",
    )
    check(
        "passive pipeline endpoint 44->38",
        f"{passive_counts['candidate_pool_stage01']}->{passive_counts['final_locked']}",
        "44->38",
    )

    # Type breakdowns: 9/13/18/11, 6/9/12/7, 5/11/26/2, 4/6/26/2.
    active_tb = active_meta["type_breakdown"]
    passive_tb = passive_meta["type_breakdown"]

    def tb_tuple(tb):
        return (tb["categorical"], tb["boolean"], tb["continuous"], tb["discrete"])

    check(
        "active candidate type breakdown (cat/bool/cont/disc)",
        tb_tuple(active_tb["candidate_stage01"]),
        (9, 13, 18, 11),
    )
    check(
        "active locked type breakdown (cat/bool/cont/disc)",
        tb_tuple(active_tb["locked_final"]),
        (6, 9, 12, 7),
    )
    check(
        "passive candidate type breakdown (cat/bool/cont/disc)",
        tb_tuple(passive_tb["candidate_stage01"]),
        (5, 11, 26, 2),
    )
    check(
        "passive locked type breakdown (cat/bool/cont/disc)",
        tb_tuple(passive_tb["locked_final"]),
        (4, 6, 26, 2),
    )

    # Stage counts: active 51->44->36->36->34->34, passive 44->50->45->39->39->38.
    active_ledger_afters = [row["count_after"] for row in active_meta["ledger"]]
    passive_ledger_afters = [row["count_after"] for row in passive_meta["ledger"]]
    check(
        "active stage-count sequence (candidate->04->05->06->09->10)",
        [51] + active_ledger_afters,
        [51, 44, 36, 36, 34, 34],
    )
    check(
        "passive stage-count sequence (candidate->04->05->06->09->10)",
        [44] + passive_ledger_afters,
        [44, 50, 45, 39, 39, 38],
    )

    # Pre-count exclusions: 16 active, 11 passive.
    check("active pre-count exclusions", active_counts["excluded_before_count"], 16)
    check("passive pre-count exclusions", passive_counts["excluded_before_count"], 11)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/dashboard/export_feature_journey.py",
        "notes": [
            "Candidate/locked feature lists parsed directly from src/eda/feature_config_v1_historical.py, "
            "feature_config_v2_historical.py and feature_config.py -- not retyped.",
            "Stage numbers/labels for every dropped or engineered candidate are looked up from hardcoded "
            "per-stage name sets, verified feature-by-feature against "
            "reports/analysis/shot_target/EDA_PIPELINE_LOG.html -- not inferred from the wording of each "
            "drop reason string.",
            "Per-feature profiles computed from ALL rows of the current (post coordinate-frame-fix, "
            "commit f7204c7 / Prompt 85) data/features/player_defensive_actions.parquet and "
            "data/features/passive_defense.parquet -- no train/test split applied here. Profiles are "
            "computed for every feature record whose column exists in the parquet, not only the 72 "
            "locked features -- dropped candidates and pre-count exclusions get one too where the "
            "column is present; profile=null with profile_reason otherwise.",
            "modelled=false on nearest_defender_distance / defenders_within_5m (active) reflects a "
            "downstream modelling-stage exclusion (train_active_binary_baseline.py), not a "
            "feature_config.py lock/drop decision -- both remain locked EDA candidates.",
            "Pre-count exclusions are labelled 'before candidate pool', not stage 0 -- they never "
            "existed as stage-01 candidates at all.",
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
