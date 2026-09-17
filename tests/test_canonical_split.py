"""Integrity tests for the frozen canonical match-grouped split (prompt 13).

These were checked manually once this session (all passed); this file makes
sure that check never silently regresses -- if outputs/models/splits/
match_assignment.json is ever regenerated, edited, or corrupted such that a
match is duplicated, dropped, or double-assigned, these tests fail loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dax.models.splits import CANONICAL_SPLIT_PATH, load_canonical_split

pytestmark = pytest.mark.skipif(
    not Path(CANONICAL_SPLIT_PATH).exists(),
    reason="Canonical split not computed in this environment (run scripts/compute_canonical_split.py).",
)

EXPECTED_TOTAL_MATCHES = 115
EXPECTED_TEST_MATCHES = 23
EXPECTED_TRAIN_VAL_MATCHES = 92
EXPECTED_N_FOLDS = 5


@pytest.fixture(scope="module")
def assignment() -> dict[str, str]:
    return load_canonical_split()


def test_every_match_accounted_for_exactly_once(assignment: dict[str, str]) -> None:
    assert len(assignment) == EXPECTED_TOTAL_MATCHES
    assert len(set(assignment.keys())) == len(assignment), "duplicate match_id keys in the assignment"


def test_labels_are_test_or_a_valid_fold(assignment: dict[str, str]) -> None:
    valid_labels = {"test"} | {f"fold{i}" for i in range(EXPECTED_N_FOLDS)}
    invalid = {mid: label for mid, label in assignment.items() if label not in valid_labels}
    assert not invalid, f"unexpected split labels: {invalid}"


def test_test_match_count(assignment: dict[str, str]) -> None:
    test_matches = [mid for mid, label in assignment.items() if label == "test"]
    assert len(test_matches) == EXPECTED_TEST_MATCHES


def test_no_match_appears_in_two_cv_folds(assignment: dict[str, str]) -> None:
    """A match_id can only map to one label at all (dict keys are unique by
    construction), but this test asserts the stronger, more specific
    property the prompt calls out explicitly: no match's fold membership is
    ambiguous or split across multiple folds."""
    fold_membership: dict[str, set[str]] = {}
    for match_id, label in assignment.items():
        if label == "test":
            continue
        fold_membership.setdefault(match_id, set()).add(label)

    multiply_assigned = {mid: folds for mid, folds in fold_membership.items() if len(folds) > 1}
    assert not multiply_assigned, f"matches assigned to more than one CV fold: {multiply_assigned}"


def test_no_test_match_appears_in_any_cv_fold(assignment: dict[str, str]) -> None:
    test_matches = {mid for mid, label in assignment.items() if label == "test"}
    fold_matches = {mid for mid, label in assignment.items() if label != "test"}
    overlap = test_matches & fold_matches
    assert not overlap, f"matches assigned to both TEST and a CV fold: {overlap}"


def test_train_val_matches_cover_all_five_folds(assignment: dict[str, str]) -> None:
    fold_labels_present = {label for label in assignment.values() if label != "test"}
    assert fold_labels_present == {f"fold{i}" for i in range(EXPECTED_N_FOLDS)}


def test_train_val_match_count(assignment: dict[str, str]) -> None:
    train_val_matches = [mid for mid, label in assignment.items() if label != "test"]
    assert len(train_val_matches) == EXPECTED_TRAIN_VAL_MATCHES


def test_fold_sizes_are_reasonably_balanced(assignment: dict[str, str]) -> None:
    """17-20 matches per fold, per the known result this session -- not a
    tight bound, just enough to catch a badly skewed regeneration."""
    from collections import Counter

    counts = Counter(label for label in assignment.values() if label != "test")
    for fold, count in counts.items():
        assert 10 <= count <= 30, f"{fold} has {count} matches, outside a sane range"
