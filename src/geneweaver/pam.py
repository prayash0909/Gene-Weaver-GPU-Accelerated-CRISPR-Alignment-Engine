
from .parser import VALID_DNA_BASES


def _is_forward_pam(sequence):
    """Check the SpCas9 forward PAM pattern NGG."""

    return (
        len(sequence) == 3
        and sequence[0] in "ACGT"
        and sequence[1:] == "GG"
    )


def _is_reverse_pam(sequence):
    """Check the reverse-strand PAM pattern CCN."""

    return (
        len(sequence) == 3
        and sequence[:2] == "CC"
        and sequence[2] in "ACGT"
    )


def find_adjacent_pams(
    genome_sequence,
    protospacer_start,
    protospacer_length
):
    """
    Find SpCas9 PAM sites adjacent to a protospacer.

    Coordinates are zero-based.
    Forward PAM: NGG immediately after the protospacer.
    Reverse PAM: CCN immediately before the protospacer.
    """

    if not isinstance(genome_sequence, str):
        raise TypeError("Genome sequence must be a string.")

    sequence = genome_sequence.upper()

    if not sequence:
        raise ValueError("Genome sequence cannot be empty.")

    if set(sequence) - VALID_DNA_BASES:
        raise ValueError("Genome contains invalid DNA bases.")

    if (
        isinstance(protospacer_start, bool)
        or not isinstance(protospacer_start, int)
    ):
        raise TypeError("Protospacer start must be an integer.")

    if (
        isinstance(protospacer_length, bool)
        or not isinstance(protospacer_length, int)
    ):
        raise TypeError("Protospacer length must be an integer.")

    if protospacer_start < 0:
        raise ValueError("Protospacer start cannot be negative.")

    if protospacer_length <= 0:
        raise ValueError("Protospacer length must be positive.")

    protospacer_end = protospacer_start + protospacer_length

    if protospacer_end > len(sequence):
        raise ValueError("Protospacer extends beyond genome sequence.")

    pams = []

    # Check forward strand: NGG after the protospacer.
    forward_pam = sequence[protospacer_end:protospacer_end + 3]

    if _is_forward_pam(forward_pam):
        pams.append({
            "position": protospacer_end,
            "strand": "+",
            "pam_sequence": forward_pam,
        })

    # Check reverse strand: CCN before the protospacer.
    reverse_start = protospacer_start - 3

    if reverse_start >= 0:
        reverse_pam = sequence[reverse_start:protospacer_start]

        if _is_reverse_pam(reverse_pam):
            pams.append({
                "position": reverse_start,
                "strand": "-",
                "pam_sequence": reverse_pam,
            })

    return pams
