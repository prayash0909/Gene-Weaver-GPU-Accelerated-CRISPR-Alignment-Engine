from pathlib import Path

from Bio import SeqIO


VALID_DNA_BASES = set("ACGTRYSWKMBDHVN")


def parse_fasta(file_path):
    """
    Parse a FASTA file and return validated DNA sequence records.

    Each record contains:
    - id
    - description
    - sequence
    - length
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records = []
    seen_ids = set()

    for record in SeqIO.parse(str(path), "fasta"):
        sequence = str(record.seq).upper()
        sequence_id = record.id

        if not sequence:
            raise ValueError(
                f"Empty DNA sequence found in record: {sequence_id}"
            )

        invalid_bases = set(sequence) - VALID_DNA_BASES

        if invalid_bases:
            invalid = "".join(sorted(invalid_bases))
            raise ValueError(
                f"Invalid DNA bases '{invalid}' in record: {sequence_id}"
            )

        if sequence_id in seen_ids:
            raise ValueError(f"Duplicate FASTA record ID: {sequence_id}")

        seen_ids.add(sequence_id)

        records.append({
            "id": sequence_id,
            "description": record.description,
            "sequence": sequence,
            "length": len(sequence),
        })

    if not records:
        raise ValueError("The FASTA file contains no valid sequence records.")

    return records

def chunk_sequence(sequence, chunk_size=1000, overlap=20):
    """
    Divide a DNA sequence into overlapping chunks.
    Avoid creating redundant trailing chunks.
    """
    if not isinstance(sequence, str):
        raise TypeError("Sequence must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not sequence:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(sequence), step):
        chunks.append(sequence[start:start + chunk_size])

        if start + chunk_size >= len(sequence):
            break

    return chunks