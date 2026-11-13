
from typing import Any, cast

import pytest

from src.geneweaver.scoring import (
    hamming_distance,
    find_approximate_matches,
)


def test_hamming_distance_with_one_mismatch():
    assert hamming_distance("ATGCGT", "ATGAGT") == 1


def test_hamming_distance_with_multiple_mismatches():
    assert hamming_distance("ATGCGT", "GTCAGT") == 3


def test_hamming_distance_with_identical_sequences():
    assert hamming_distance("ATGC", "ATGC") == 0


def test_hamming_distance_rejects_different_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        hamming_distance("ATGC", "ATG")


def test_hamming_distance_rejects_invalid_bases():
    with pytest.raises(ValueError, match="invalid DNA"):
        hamming_distance("ATGX", "ATGC")


def test_hamming_distance_rejects_non_string():
    with pytest.raises(TypeError):
        hamming_distance(123, "ATGC")


def test_approximate_match_with_one_mismatch():
    records = [{"id": "gene_001", "sequence": "ATGAGT"}]

    matches = find_approximate_matches("ATGCGT", records)

    assert len(matches) == 1
    assert matches[0]["mismatches"] == 1
    assert matches[0]["position"] == 0


def test_mismatch_positions_are_reported():
    records = [{"id": "gene_001", "sequence": "ATGAGT"}]

    matches = find_approximate_matches("ATGCGT", records)

    assert matches[0]["mismatch_positions"] == [3]


def test_maximum_mismatch_limit():
    records = [
        {"id": "gene_001", "sequence": "ATGAGT"},
        {"id": "gene_002", "sequence": "ATGACT"},
    ]

    matches = find_approximate_matches(
        "ATGCGT", records, max_mismatches=1
    )

    assert len(matches) == 1
    assert matches[0]["record_id"] == "gene_001"


def test_exact_match_is_also_detected():
    records = [{"id": "gene_001", "sequence": "ATGCGT"}]

    matches = find_approximate_matches(
        "ATGCGT", records, max_mismatches=0
    )

    assert len(matches) == 1
    assert matches[0]["mismatches"] == 0


def test_overlapping_approximate_matches():
    records = [{"id": "gene_001", "sequence": "AAAAA"}]

    matches = find_approximate_matches(
        "AAA", records, max_mismatches=0
    )

    assert [match["position"] for match in matches] == [0, 1, 2]


def test_case_insensitive_approximate_matching():
    records = [{"id": "gene_001", "sequence": "atgagt"}]

    matches = find_approximate_matches("ATGCGT", records)

    assert len(matches) == 1
    assert matches[0]["matched_sequence"] == "ATGAGT"


def test_negative_mismatch_limit_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        find_approximate_matches("ATGC", [], max_mismatches=-1)


def test_non_integer_mismatch_limit_is_rejected():
    with pytest.raises(TypeError, match="must be an integer"):
        find_approximate_matches(
            "ATGC", [], max_mismatches=cast(Any, 1.5)
        )


def test_empty_target_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        find_approximate_matches("", [])


def test_target_longer_than_genome_returns_no_match():
    records = [{"id": "gene_001", "sequence": "ATG"}]

    assert find_approximate_matches("ATGC", records) == []
