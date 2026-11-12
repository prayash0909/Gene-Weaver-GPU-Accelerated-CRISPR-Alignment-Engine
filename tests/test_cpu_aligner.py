import pytest

from src.geneweaver.cpu_aligner import find_exact_matches


def test_single_exact_match():
    records = [{"id": "gene_001", "sequence": "GGATGCTT"}]

    matches = find_exact_matches("ATGC", records)

    assert len(matches) == 1
    assert matches[0]["position"] == 2
    assert matches[0]["record_id"] == "gene_001"


def test_multiple_exact_matches():
    records = [{"id": "gene_001", "sequence": "ATGCATGC"}]

    matches = find_exact_matches("ATGC", records)

    assert [match["position"] for match in matches] == [0, 4]


def test_overlapping_matches():
    records = [{"id": "gene_001", "sequence": "AAAAA"}]

    matches = find_exact_matches("AAA", records)

    assert [match["position"] for match in matches] == [0, 1, 2]


def test_multiple_genome_records():
    records = [
        {"id": "gene_001", "sequence": "GGATGC"},
        {"id": "gene_002", "sequence": "ATGCCC"},
    ]

    matches = find_exact_matches("ATGC", records)

    assert len(matches) == 2
    assert matches[0]["record_id"] == "gene_001"
    assert matches[1]["record_id"] == "gene_002"


def test_matching_is_case_insensitive():
    records = [{"id": "gene_001", "sequence": "ggatgc"}]

    matches = find_exact_matches("atgc", records)

    assert len(matches) == 1
    assert matches[0]["matched_sequence"] == "ATGC"


def test_no_match_returns_empty_list():
    records = [{"id": "gene_001", "sequence": "GGGGCCCC"}]

    assert find_exact_matches("ATGC", records) == []


def test_empty_target_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        find_exact_matches("", [])


def test_invalid_target_bases_are_rejected():
    records = [{"id": "gene_001", "sequence": "ATGC"}]

    with pytest.raises(ValueError, match="Invalid target"):
        find_exact_matches("ATGX", records)


def test_invalid_genome_bases_are_rejected():
    records = [{"id": "gene_001", "sequence": "ATGX"}]

    with pytest.raises(ValueError, match="Invalid genome"):
        find_exact_matches("ATGC", records)


def test_missing_record_fields_are_rejected():
    records = [{"id": "gene_001"}]

    with pytest.raises(ValueError, match="must contain id and sequence"):
        find_exact_matches("ATGC", records)


def test_target_longer_than_genome_returns_no_match():
    records = [{"id": "gene_001", "sequence": "ATG"}]

    assert find_exact_matches("ATGC", records) == []
