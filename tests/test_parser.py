import pytest

from src.geneweaver.parser import parse_fasta


def test_parse_multiple_fasta_records(tmp_path):
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(
        ">gene_001 Human Target\n"
        "ATGCGT\n"
        ">gene_002 Off Target\n"
        "GGGCCC\n"
    )

    records = parse_fasta(fasta_file)

    assert len(records) == 2
    assert records[0]["id"] == "gene_001"
    assert records[0]["sequence"] == "ATGCGT"
    assert records[0]["length"] == 6


def test_sequence_is_converted_to_uppercase(tmp_path):
    fasta_file = tmp_path / "lowercase.fasta"
    fasta_file.write_text(">gene_001\natgcgt\n")

    records = parse_fasta(fasta_file)

    assert records[0]["sequence"] == "ATGCGT"


def test_invalid_dna_bases_are_rejected(tmp_path):
    fasta_file = tmp_path / "invalid.fasta"
    fasta_file.write_text(">gene_001\nATGCXYZ\n")

    with pytest.raises(ValueError, match="Invalid DNA bases"):
        parse_fasta(fasta_file)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_fasta(tmp_path / "missing.fasta")


def test_empty_fasta_file_is_rejected(tmp_path):
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    with pytest.raises(ValueError, match="no valid sequence records"):
        parse_fasta(fasta_file)


def test_duplicate_record_ids_are_rejected(tmp_path):
    fasta_file = tmp_path / "duplicate.fasta"
    fasta_file.write_text(
        ">gene_001\nATGC\n"
        ">gene_001\nGGCC\n"
    )

    with pytest.raises(ValueError, match="Duplicate FASTA record ID"):
        parse_fasta(fasta_file)

from src.geneweaver.parser import chunk_sequence


def test_sequence_is_split_into_chunks():
    sequence = "A" * 25

    chunks = chunk_sequence(sequence, chunk_size=10, overlap=2)

    assert chunks == ["A" * 10, "A" * 10, "A" * 9]


def test_chunking_preserves_original_sequence():
    sequence = "ATGCGTACGTAGCTAGCTAG"

    chunks = chunk_sequence(sequence, chunk_size=8, overlap=2)

    reconstructed = chunks[0] + "".join(
        chunk[2:] for chunk in chunks[1:]
    )

    assert reconstructed == sequence


def test_empty_sequence_returns_empty_list():
    assert chunk_sequence("") == []


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=5, overlap=5)
