
from .parser import VALID_DNA_BASES


def hamming_distance(sequence_a, sequence_b):
    """Count the number of different positions in two DNA sequences."""

    if not isinstance(sequence_a, str) or not isinstance(sequence_b, str):
        raise TypeError("Both sequences must be strings.")

    if len(sequence_a) != len(sequence_b):
        raise ValueError("Sequences must have equal lengths.")

    sequence_a = sequence_a.upper()
    sequence_b = sequence_b.upper()

    invalid_a = set(sequence_a) - VALID_DNA_BASES
    invalid_b = set(sequence_b) - VALID_DNA_BASES

    if invalid_a or invalid_b:
        raise ValueError("Sequences contain invalid DNA bases.")

    return sum(
        base_a != base_b
        for base_a, base_b in zip(sequence_a, sequence_b)
    )


def find_approximate_matches(
    target_sequence,
    genome_records,
    max_mismatches=2
):
    """Find genome regions within the allowed mismatch limit."""

    if not isinstance(target_sequence, str):
        raise TypeError("Target sequence must be a string.")

    target = target_sequence.upper()

    if not target:
        raise ValueError("Target sequence cannot be empty.")

    if set(target) - VALID_DNA_BASES:
        raise ValueError("Target contains invalid DNA bases.")

    if isinstance(max_mismatches, bool) or not isinstance(max_mismatches, int):
        raise TypeError("Maximum mismatches must be an integer.")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative.")

    if not isinstance(genome_records, list):
        raise TypeError("Genome records must be a list.")

    matches = []
    target_length = len(target)

    for record in genome_records:
        if not isinstance(record, dict):
            raise TypeError("Each genome record must be a dictionary.")

        if "id" not in record or "sequence" not in record:
            raise ValueError("Each record must contain id and sequence.")

        sequence = record["sequence"]

        if not isinstance(sequence, str):
            raise TypeError("Genome sequence must be a string.")

        sequence = sequence.upper()

        if not sequence:
            raise ValueError(f"Empty genome sequence: {record['id']}")

        if set(sequence) - VALID_DNA_BASES:
            raise ValueError(
                f"Invalid genome DNA bases in record: {record['id']}"
            )

        for position in range(len(sequence) - target_length + 1):
            candidate = sequence[position:position + target_length]

            distance = hamming_distance(target, candidate)

            if distance <= max_mismatches:
                mismatch_positions = [
                    index
                    for index, (a, b) in enumerate(zip(target, candidate))
                    if a != b
                ]

                matches.append({
                    "record_id": record["id"],
                    "position": position,
                    "matched_sequence": candidate,
                    "mismatches": distance,
                    "mismatch_positions": mismatch_positions,
                })

    return matches
