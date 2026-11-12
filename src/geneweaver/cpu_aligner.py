from .parser import VALID_DNA_BASES


def find_exact_matches(target_sequence, genome_records):
    """
    Find all exact occurrences of a target DNA sequence.
    Supports multiple genome records and overlapping matches.
    """

    if not isinstance(target_sequence, str):
        raise TypeError("Target sequence must be a string.")

    target = target_sequence.upper()

    if not target:
        raise ValueError("Target sequence cannot be empty.")

    invalid_bases = set(target) - VALID_DNA_BASES

    if invalid_bases:
        invalid = "".join(sorted(invalid_bases))
        raise ValueError(f"Invalid target DNA bases: {invalid}")

    if not isinstance(genome_records, list):
        raise TypeError("Genome records must be a list.")

    matches = []

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

        invalid_genome_bases = set(sequence) - VALID_DNA_BASES

        if invalid_genome_bases:
            invalid = "".join(sorted(invalid_genome_bases))
            raise ValueError(
                f"Invalid genome DNA bases '{invalid}' "
                f"in record: {record['id']}"
            )

        start = 0

        while True:
            position = sequence.find(target, start)

            if position == -1:
                break

            matches.append({
                "record_id": record["id"],
                "position": position,
                "matched_sequence": target,
            })

            start = position + 1

    return matches