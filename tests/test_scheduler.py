
import pytest

from geneweaver.cpu_aligner import find_exact_matches
from geneweaver.scheduler import find_exact_matches_parallel


@pytest.fixture
def genome_records():
    return [
        {
            "id": "gene_001",
            "sequence": "ATGCGTACGTAGCTAGCTAG",
        },
        {
            "id": "gene_002",
            "sequence": "ATGCGTACGTAGATAGCTAG",
        },
        {
            "id": "gene_003",
            "sequence": "GGGAAACCCGGGTTTAAACC",
        },
    ]


def test_parallel_results_match_cpu_results(genome_records):
    target = "ATGCGTACGTAGCTAGCTAG"

    expected = find_exact_matches(target, genome_records)
    actual = find_exact_matches_parallel(target, genome_records)

    assert actual == expected

def test_parallel_results_preserve_record_order(genome_records):
    target = "ATG"

    results = find_exact_matches_parallel(target, genome_records)

    record_ids = [match["record_id"] for match in results]

    assert record_ids == [
        "gene_001",
        "gene_002",
    ]


def test_parallel_matching_with_custom_workers(genome_records):
    target = "ATG"

    results = find_exact_matches_parallel(
        target,
        genome_records,
        num_workers=2,
    )

    assert len(results) == 2


def test_parallel_matching_with_empty_records():
    assert find_exact_matches_parallel("ATG", []) == []


def test_parallel_matching_rejects_invalid_worker_count(genome_records):
    with pytest.raises(ValueError):
        find_exact_matches_parallel(
            "ATG",
            genome_records,
            num_workers=0,
        )


def test_parallel_matching_rejects_non_integer_workers(genome_records):
    with pytest.raises(TypeError):
        find_exact_matches_parallel(
            "ATG",
            genome_records,
            num_workers=2.5,
        )
def test_parallel_matching_rejects_non_list_genome_records():
    with pytest.raises(TypeError, match="Genome records must be a list."):
        find_exact_matches_parallel("ATG", "ATGCGT")


def test_parallel_matching_rejects_invalid_genome_record():
    invalid_records = [
        {
            "id": "gene_001",
            "sequence": 12345,
        }
    ]

    with pytest.raises(TypeError):
        find_exact_matches_parallel("ATG", invalid_records)