
from typing import Any, cast

import pytest

from numba import cuda

from geneweaver.gpu_aligner import find_approximate_matches_accelerated
from geneweaver.scoring import find_approximate_matches


def test_cpu_fallback_matches_scoring_results():
    records = [
        {"id": "gene_001", "sequence": "ATGCGTACGTAGCTAGCTAG"},
        {"id": "gene_002", "sequence": "ATGCGTACGTAGATAGCTAG"},
    ]

    expected = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=2,
    )

    actual = find_approximate_matches_accelerated(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=2,
    )

    assert actual == expected


def test_all_iupac_bases_are_supported():
    records = [
        {"id": "gene_001", "sequence": "ATGNRYSWKMBDHVATG"}
    ]

    expected = find_approximate_matches(
        "ATGNRYS",
        records,
        max_mismatches=0,
    )

    actual = find_approximate_matches_accelerated(
        "ATGNRYS",
        records,
        max_mismatches=0,
    )

    assert actual == expected


def test_empty_target_is_rejected():
    with pytest.raises(ValueError):
        find_approximate_matches_accelerated("", [])


def test_invalid_target_base_is_rejected():
    with pytest.raises(ValueError):
        find_approximate_matches_accelerated("ATGX", [])


def test_non_string_target_is_rejected():
    with pytest.raises(TypeError):
        find_approximate_matches_accelerated(None, [])


def test_negative_mismatch_limit_is_rejected():
    with pytest.raises(ValueError):
        find_approximate_matches_accelerated("ATGC", [], -1)


def test_non_list_genome_records_are_rejected():
    with pytest.raises(TypeError):
        find_approximate_matches_accelerated("ATGC", ())


def test_invalid_genome_sequence_type_is_rejected():
    records = [{"id": "gene_001", "sequence": 123}]

    with pytest.raises(TypeError):
        find_approximate_matches_accelerated("ATGC", records)


def test_short_genome_sequence_is_skipped():
    records = [{"id": "gene_001", "sequence": "AT"}]

    result = find_approximate_matches_accelerated("ATGC", records)

    assert result == []


def test_pam_results_match_scoring_module():
    records = [
        {"id": "gene_001", "sequence": "ATGCGTACGTAGCTAGCTAGAGG"}
    ]

    expected = find_approximate_matches(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0,
        include_pam=True,
    )

    actual = find_approximate_matches_accelerated(
        "ATGCGTACGTAGCTAGCTAG",
        records,
        max_mismatches=0,
        include_pam=True,
    )

    assert actual == expected


def test_gpu_request_fails_when_cuda_unavailable():
    if cuda.is_available():
        pytest.skip("CUDA is available on this system.")

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        find_approximate_matches_accelerated(
            "ATGC",
            [{"id": "gene_001", "sequence": "ATGC"}],
            use_gpu=True,
        )


def test_invalid_gpu_option_is_rejected():
    with pytest.raises(TypeError):
        find_approximate_matches_accelerated(
            "ATGC",
            [],
            use_gpu=cast(Any, "yes"),
        )
