import pytest

from src.geneweaver.pam import find_adjacent_pams


def test_detects_forward_ngg_pam():
    result = find_adjacent_pams(
        "ATGCGTACGTAGCTAGCTAGAGG",
        0,
        20
    )

    assert result == [
        {
            "position": 20,
            "strand": "+",
            "pam_sequence": "AGG",
        }
    ]


def test_detects_reverse_ccn_pam():
    result = find_adjacent_pams(
        "CCCATGCGTACGTAGCTAGCTAG",
        3,
        20
    )

    assert result == [
        {
            "position": 0,
            "strand": "-",
            "pam_sequence": "CCC",
        }
    ]


def test_returns_empty_when_no_pam_exists():
    result = find_adjacent_pams(
        "ATGCGTACGTAGCTAGCTAGAAA",
        0,
        20
    )

    assert result == []

def test_detects_pams_on_both_strands():
    result = find_adjacent_pams(
        "CCCATGCGTACGTAGCTAGCTAGAGG",
        3,
        20
    )

    assert len(result) == 2

    strands = {pam["strand"] for pam in result}

    assert strands == {"+", "-"}

    pam_sequences = {pam["pam_sequence"] for pam in result}

    assert pam_sequences == {"CCC", "AGG"}


def test_pam_detection_is_case_insensitive():
    result = find_adjacent_pams(
        "atgcgtacgtagctagctagagg",
        0,
        20
    )

    assert result[0]["pam_sequence"] == "AGG"


def test_rejects_invalid_dna_sequence():
    with pytest.raises(ValueError):
        find_adjacent_pams("ATGCGTACGTAGCTAGCTAGXGG", 0, 20)


def test_rejects_negative_start_position():
    with pytest.raises(ValueError):
        find_adjacent_pams("ATGCGTACGTAGCTAGCTAGAGG", -1, 20)


def test_rejects_zero_protospacer_length():
    with pytest.raises(ValueError):
        find_adjacent_pams("ATGCGTACGTAGCTAGCTAGAGG", 0, 0)


def test_rejects_protospacer_outside_genome():
    with pytest.raises(ValueError):
        find_adjacent_pams("ATGCGT", 0, 20)


def test_rejects_non_integer_start_position():
    with pytest.raises(TypeError):
        find_adjacent_pams("ATGCGTACGTAGCTAGCTAGAGG", 0.5, 20)