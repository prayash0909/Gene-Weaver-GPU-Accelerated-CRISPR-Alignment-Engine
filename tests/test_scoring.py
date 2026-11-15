
from typing import Any, cast
from src.geneweaver.scoring import _hamming_distance_with_limit

import pytest

from src.geneweaver.scoring import (
    hamming_distance,
    find_approximate_matches,
    rank_crispr_candidates,
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
def test_early_exit_returns_none_when_limit_is_exceeded():
    result = _hamming_distance_with_limit("AAAA", "TTTT", 1)
    assert result is None


def test_early_exit_returns_mismatch_count_within_limit():
    result = _hamming_distance_with_limit("AAAA", "AATA", 1)
    assert result == 1
def test_approximate_matching_includes_forward_pam():
    records = [
        {
            "id": "gene_001",
            "sequence": "ATGCGTACGTAGCTAGCTAGAGG"
        }
    ]

    matches = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0,
        include_pam=True
    )

    assert len(matches) == 1
    assert matches[0]["pam_found"] is True
    assert matches[0]["pams"][0]["strand"] == "+"
    assert matches[0]["pams"][0]["pam_sequence"] == "AGG"


def test_approximate_matching_includes_reverse_pam():
    records = [
        {
            "id": "gene_002",
            "sequence": "CCCATGCGTACGTAGCTAGCTAG"
        }
    ]

    matches = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0,
        include_pam=True
    )

    assert len(matches) == 1
    assert matches[0]["pam_found"] is True
    assert matches[0]["pams"][0]["strand"] == "-"


def test_approximate_matching_reports_missing_pam():
    records = [
        {
            "id": "gene_003",
            "sequence": "ATGCGTACGTAGCTAGCTAGAAA"
        }
    ]

    matches = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0,
        include_pam=True
    )

    assert len(matches) == 1
    assert matches[0]["pam_found"] is False
    assert matches[0]["pams"] == []


def test_pam_is_not_included_by_default():
    records = [
        {
            "id": "gene_001",
            "sequence": "ATGCGTACGTAGCTAGCTAGAGG"
        }
    ]

    matches = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0
    )

    assert "pam_found" not in matches[0]
    assert "pams" not in matches[0]


def test_rank_candidates_prioritizes_pam_and_fewer_mismatches():
    matches = [
        {"pam_found": False, "mismatches": 0},
        {"pam_found": True, "mismatches": 2},
        {"pam_found": True, "mismatches": 0},
        {"pam_found": True, "mismatches": 1},
    ]

    ranked = rank_crispr_candidates(matches)

    assert [match["mismatches"] for match in ranked] == [0, 1, 2, 0]
    assert [match["pam_found"] for match in ranked] == [
        True, True, True, False
    ]
    assert [match["rank"] for match in ranked] == [1, 2, 3, 4]


def test_rank_candidates_rejects_missing_pam_information():
    with pytest.raises(ValueError):
        rank_crispr_candidates([
            {"mismatches": 1}
        ])