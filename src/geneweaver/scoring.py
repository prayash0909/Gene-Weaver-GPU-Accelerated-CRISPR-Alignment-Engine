
from .parser import VALID_DNA_BASES
from .pam import find_adjacent_pams


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


def _hamming_distance_with_limit(
    sequence_a,
    sequence_b,
    max_mismatches
):
    """Stop counting when the mismatch limit is exceeded."""

    mismatches = 0

    for base_a, base_b in zip(sequence_a, sequence_b):
        if base_a != base_b:
            mismatches += 1

            if mismatches > max_mismatches:
                return None

    return mismatches


def find_approximate_matches(
    target_sequence,
    genome_records,
    max_mismatches=2,
    include_pam=False
):
    """Find approximate matches and optionally annotate PAM sites."""

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

    if not isinstance(include_pam, bool):
        raise TypeError("include_pam must be a boolean.")

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

            distance = _hamming_distance_with_limit(
                target,
                candidate,
                max_mismatches
            )

            if distance is not None:
                mismatch_positions = [
                    index
                    for index, (a, b) in enumerate(zip(target, candidate))
                    if a != b
                ]

                match = {
                    "record_id": record["id"],
                    "position": position,
                    "matched_sequence": candidate,
                    "mismatches": distance,
                    "mismatch_positions": mismatch_positions,
                }

                if include_pam:
                    pams = find_adjacent_pams(
                        sequence,
                        position,
                        target_length
                    )

                    match["pams"] = pams
                    match["pam_found"] = bool(pams)

                matches.append(match)

    return matches


def rank_crispr_candidates(matches):
    """Rank candidates by PAM availability and mismatch count."""

    if not isinstance(matches, list):
        raise TypeError("Matches must be a list.")

    for match in matches:
        if not isinstance(match, dict):
            raise TypeError("Each match must be a dictionary.")

        if "pam_found" not in match or "mismatches" not in match:
            raise ValueError(
                "Each match must contain pam_found and mismatches."
            )

        if not isinstance(match["pam_found"], bool):
            raise TypeError("pam_found must be a boolean.")

        if (
            isinstance(match["mismatches"], bool)
            or not isinstance(match["mismatches"], int)
            or match["mismatches"] < 0
        ):
            raise ValueError("Mismatches must be a non-negative integer.")

    ranked = sorted(
        matches,
        key=lambda match: (
            not match["pam_found"],
            match["mismatches"]
        )
    )

    results = []

    for rank, match in enumerate(ranked, start=1):
        ranked_match = match.copy()
        ranked_match["rank"] = rank
        results.append(ranked_match)

    return results
