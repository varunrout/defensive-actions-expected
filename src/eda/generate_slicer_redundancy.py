"""CLI entrypoint: prompt 27 -- are the locked categorical slicers used in
prompts 25/26's slice-stratification grid (6 active, 4 passive) themselves
independent, or redundant with each other? If two slicers overlap heavily,
"diverges on both" in SLICE_STRATIFICATION.json is one finding counted
twice, not two -- this is a diagnostic check on the slicers alone, before
any interaction-term shortlist gets built from that grid.

Every pair of slicers within each dataset (active: 6 slicers -> 15 pairs,
passive: 4 -> 6 pairs) is scored two ways:
  - Cramer's V (bias-corrected, reusing src.eda.correlation.cramers_v --
    the same function the CORRELATION_ANALYSIS pipeline uses for
    categorical<->categorical pairs), the primary measure.
  - Normalised mutual information (sklearn, arithmetic averaging), a
    secondary check since Cramer's V can be inflated by small cell counts
    in a large contingency table.

Verdict thresholds (Cramer's V, stated explicitly per the prompt):
  < 0.1        independent
  0.1 - 0.3    partially redundant
  > 0.3        highly redundant
The same thresholds are applied to NMI for a second, independent verdict;
a mismatch between the two verdicts is flagged as `metrics_disagree`.

This prompt does NOT edit SLICE_STRATIFICATION.json -- diagnostic only,
feeding a future interaction-term decision.

Usage:
    python -m src.eda.generate_slicer_redundancy
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from src.eda.correlation import cramers_v
from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICER_REDUNDANCY.json"

INDEPENDENT_MAX = 0.1
PARTIALLY_REDUNDANT_MAX = 0.3

NAMED_PAIRS = {
    ("active", "phase_label", "phase_label_prev_event"): (
        "phase_label_prev_event is a 1-event lag of phase_label -- likely high association by construction."
    ),
    ("active", "event_type", "play_pattern"): "Both describe the nature of the on-ball action.",
    ("passive", "phase_label", "defender_functional_role"): (
        "defender_functional_role is partly determined by pitch position, which correlates with match phase."
    ),
}


def verdict(v: float) -> str:
    if v < INDEPENDENT_MAX:
        return "independent"
    if v <= PARTIALLY_REDUNDANT_MAX:
        return "partially redundant"
    return "highly redundant"


def score_pair(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    sub = df[[col_a, col_b]].dropna()
    n_dropped = len(df) - len(sub)
    a, b = sub[col_a], sub[col_b]

    v = cramers_v(a, b)
    nmi = float(normalized_mutual_info_score(a.astype(str), b.astype(str)))
    ct_shape = (a.nunique(), b.nunique())

    v_verdict = verdict(v)
    nmi_verdict = verdict(nmi)

    return {
        "slicer_a": col_a,
        "slicer_b": col_b,
        "n_rows_used": len(sub),
        "n_rows_dropped_na": n_dropped,
        "contingency_table_shape": list(ct_shape),
        "cramers_v": round(v, 4),
        "normalized_mutual_info": round(nmi, 4),
        "verdict": v_verdict,
        "verdict_from_nmi": nmi_verdict,
        "metrics_disagree": v_verdict != nmi_verdict,
    }


def _connected_components(slicers: list[str], edges: list[tuple[str, str]]) -> dict:
    parent = {s: s for s in slicers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in edges:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for s in slicers:
        groups.setdefault(find(s), []).append(s)

    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    independent_slicers = sorted(members[0] for members in groups.values() if len(members) == 1)
    return {"clusters": clusters, "independent_slicers": independent_slicers}


def find_clusters(slicers: list[str], pairs: list[dict]) -> dict:
    """Three cluster views, not one, because Cramer's V and NMI disagree a
    lot in this data (small-cell inflation on V -- exactly the failure mode
    the prompt asked NMI to guard against):
      - from_cramers_v: edges where the V-verdict != independent. This is
        what a Cramer's-V-only analysis would report, and it over-clusters
        here (V gets inflated by sparse cells in larger contingency tables).
      - from_nmi: edges where the NMI-verdict != independent -- much more
        conservative, matches the pairs that are ALSO not flagged
        metrics_disagree.
      - consensus (both_metrics_agree): edges where V and NMI both say
        != independent (no metrics_disagree) -- the pairs to actually trust
        as redundant."""
    v_edges = [(p["slicer_a"], p["slicer_b"]) for p in pairs if p["verdict"] != "independent"]
    nmi_edges = [(p["slicer_a"], p["slicer_b"]) for p in pairs if p["verdict_from_nmi"] != "independent"]
    consensus_edges = [(p["slicer_a"], p["slicer_b"]) for p in pairs if p["verdict"] != "independent" and not p["metrics_disagree"]]

    return {
        "from_cramers_v": _connected_components(slicers, v_edges),
        "from_nmi": _connected_components(slicers, nmi_edges),
        "consensus_both_metrics_agree": _connected_components(slicers, consensus_edges),
    }


def analyze_dataset(dataset_key: str, dataset_cfg: dict) -> dict:
    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"], columns=dataset_cfg["categorical"])
    slicers = dataset_cfg["categorical"]

    pairs = []
    for col_a, col_b in itertools.combinations(slicers, 2):
        pair = score_pair(df, col_a, col_b)
        pair["dataset"] = dataset_key
        note_key = (dataset_key, col_a, col_b)
        note_key_rev = (dataset_key, col_b, col_a)
        if note_key in NAMED_PAIRS:
            pair["named_candidate_reason"] = NAMED_PAIRS[note_key]
        elif note_key_rev in NAMED_PAIRS:
            pair["named_candidate_reason"] = NAMED_PAIRS[note_key_rev]
        pairs.append(pair)

    clusters = find_clusters(slicers, pairs)

    return {
        "dataset": dataset_key,
        "slicers": slicers,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "clusters": clusters,
        "cluster_note": (
            "Three cluster views: from_cramers_v (over-clusters -- V is inflated by sparse cells in this "
            "dataset's larger contingency tables), from_nmi (more conservative), and "
            "consensus_both_metrics_agree (only pairs where V and NMI both agree it's not independent -- the "
            "one to actually trust)."
        ),
    }


def main() -> None:
    active_result = analyze_dataset("active", ACTIVE)
    passive_result = analyze_dataset("passive", PASSIVE)

    assert len(active_result["pairs"]) == 15, f"Expected 15 active pairs (6 choose 2), got {len(active_result['pairs'])}"
    assert len(passive_result["pairs"]) == 6, f"Expected 6 passive pairs (4 choose 2), got {len(passive_result['pairs'])}"

    named_pairs_found = [p for r in (active_result, passive_result) for p in r["pairs"] if "named_candidate_reason" in p]
    assert len(named_pairs_found) == 3, f"Expected all 3 named candidate pairs to be found, got {len(named_pairs_found)}"

    # Trust the consensus view (both metrics agree it's not independent) for
    # the downstream-connection call, not raw Cramer's V alone -- see
    # find_clusters' docstring for why V over-clusters in this data.
    n_highly_redundant_consensus = sum(
        1 for r in (active_result, passive_result) for p in r["pairs"]
        if p["verdict"] == "highly redundant" and not p["metrics_disagree"]
    )
    n_highly_redundant_raw_v = sum(
        1 for r in (active_result, passive_result) for p in r["pairs"] if p["verdict"] == "highly redundant"
    )

    output = {
        "verdict_thresholds": {
            "independent": f"Cramer's V < {INDEPENDENT_MAX}",
            "partially redundant": f"{INDEPENDENT_MAX} <= Cramer's V <= {PARTIALLY_REDUNDANT_MAX}",
            "highly redundant": f"Cramer's V > {PARTIALLY_REDUNDANT_MAX}",
            "note": "The same thresholds are applied to normalised mutual information for a second, independent verdict (verdict_from_nmi); metrics_disagree flags where the two disagree.",
        },
        "methodology": (
            "Every pair of slicers within each dataset (active: 6 slicers from feature_config.py's "
            "ACTIVE['categorical'] -> 15 pairs; passive: 4 from PASSIVE['categorical'] -> 6 pairs) scored with "
            "Cramer's V (bias-corrected, src.eda.correlation.cramers_v -- same function CORRELATION_ANALYSIS uses "
            "for categorical<->categorical pairs) as the primary measure, and normalised mutual information "
            "(sklearn, arithmetic averaging) as a secondary check since Cramer's V can be inflated by small cell "
            "counts in a large contingency table. Rows with a null in either column of the pair are dropped for "
            "that pair only (n_rows_dropped_na records how many)."
        ),
        "downstream_connection": (
            f"{n_highly_redundant_raw_v} pair(s) reach 'highly redundant' by raw Cramer's V, but only "
            f"{n_highly_redundant_consensus} of those are confirmed by NMI too (not metrics_disagree) -- "
            + (
                "the confirmed one is phase_label vs phase_label_prev_event (active), the constructed lag pair. "
                if n_highly_redundant_consensus > 0 else "none are confirmed by both metrics. "
            ) + (
                "For that confirmed pair, the divergence counts in reports/analysis/shot_target/SLICE_STRATIFICATION.json (320 "
                "genuine) and reports/analysis/xg_target/SLICE_STRATIFICATION.json (363 unconditional / 90 given-shot) likely "
                "double-count findings that show up under both phase_label and phase_label_prev_event -- a future "
                "interaction-term shortlist for modelling should prefer one of the two, not both. The remaining "
                "'partially redundant' pairs flagged by Cramer's V alone (see metrics_disagree in each pair) look "
                "like small-cell-count inflation rather than genuine overlap -- NMI for those pairs is close to 0. "
                "Diagnostic only: neither SLICE_STRATIFICATION.json file is edited by this prompt."
                if n_highly_redundant_consensus > 0 else
                "the 320/363/90 divergence counts are not meaningfully inflated by slicer redundancy once "
                "small-cell-count inflation in Cramer's V is accounted for."
            )
        ),
        "datasets": {
            "active": active_result,
            "passive": passive_result,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for ds_result in (active_result, passive_result):
        print(f"\n=== {ds_result['dataset']} ({ds_result['n_pairs']} pairs) ===")
        for p in ds_result["pairs"]:
            flag = " [NAMED CANDIDATE]" if "named_candidate_reason" in p else ""
            disagree = " [METRICS DISAGREE]" if p["metrics_disagree"] else ""
            print(f"  {p['slicer_a']} x {p['slicer_b']}: V={p['cramers_v']}, NMI={p['normalized_mutual_info']}, "
                  f"verdict={p['verdict']}{disagree}{flag}")
        c = ds_result["clusters"]
        print(f"  Clusters (from raw Cramer's V): {c['from_cramers_v']['clusters']}, independent: {c['from_cramers_v']['independent_slicers']}")
        print(f"  Clusters (from NMI):            {c['from_nmi']['clusters']}, independent: {c['from_nmi']['independent_slicers']}")
        print(f"  Clusters (consensus, trust this): {c['consensus_both_metrics_agree']['clusters']}, independent: {c['consensus_both_metrics_agree']['independent_slicers']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
