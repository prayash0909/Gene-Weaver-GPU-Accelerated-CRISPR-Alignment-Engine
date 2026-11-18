
import numpy as np
from numba import cuda, njit

from .parser import VALID_DNA_BASES
from .pam import find_adjacent_pams

BASE_TO_INT = {
    base: index
    for index, base in enumerate("ABCDGHKMNRSTVWY")
}

@njit
def _cpu_alignment_kernel(
    genome,
    target,
    max_mismatches
):
    """Perform approximate DNA matching using CPU."""

    genome_length = len(genome)
    target_length = len(target)

    match_positions = []
    mismatch_counts = []

    for position in range(genome_length - target_length + 1):
        mismatches = 0

        for index in range(target_length):
            if genome[position + index] != target[index]:
                mismatches += 1

                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            match_positions.append(position)
            mismatch_counts.append(mismatches)

    return match_positions, mismatch_counts


@cuda.jit
def _gpu_alignment_kernel(
    genome,
    target,
    max_mismatches,
    match_flags,
    mismatch_counts
):
    """Compare one genome position per CUDA thread."""

    position = cuda.grid(1)  # type: ignore[reportCallIssue]

    target_length = target.size
    total_positions = genome.size - target_length + 1

    if position < total_positions:
        mismatches = 0

        for index in range(target_length):
            if genome[position + index] != target[index]:
                mismatches += 1

        mismatch_counts[position] = mismatches

        if mismatches <= max_mismatches:
            match_flags[position] = 1


def _encode_sequence(sequence):
    """Convert DNA bases into numeric values."""

    return np.array(
        [BASE_TO_INT[base] for base in sequence],
        dtype=np.uint8
    )


def find_approximate_matches_accelerated(
    target_sequence,
    genome_records,
    max_mismatches=2,
    include_pam=False,
    use_gpu=False
):
    """
    Find approximate DNA matches using CPU or CUDA.

    CUDA execution requires a compatible NVIDIA GPU.
    CPU fallback is used when use_gpu is False.
    """

    if not isinstance(target_sequence, str):
        raise TypeError("Target sequence must be a string.")

    target = target_sequence.upper()

    if not target:
        raise ValueError("Target sequence cannot be empty.")

    if set(target) - VALID_DNA_BASES:
        raise ValueError("Target contains invalid DNA bases.")

    if (
        isinstance(max_mismatches, bool)
        or not isinstance(max_mismatches, int)
    ):
        raise TypeError("Maximum mismatches must be an integer.")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative.")

    if not isinstance(genome_records, list):
        raise TypeError("Genome records must be a list.")

    if not isinstance(include_pam, bool):
        raise TypeError("include_pam must be a boolean.")

    if not isinstance(use_gpu, bool):
        raise TypeError("use_gpu must be a boolean.")

    if use_gpu and not cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    encoded_target = _encode_sequence(target)
    target_length = len(target)

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

        if set(sequence) - VALID_DNA_BASES:
            raise ValueError(
                f"Invalid genome DNA bases in record: {record['id']}"
            )

        if len(sequence) < target_length:
            continue

        encoded_genome = _encode_sequence(sequence)

        if use_gpu:
            total_positions = len(sequence) - target_length + 1

            device_genome = cuda.to_device(encoded_genome)
            device_target = cuda.to_device(encoded_target)

            match_flags = cuda.to_device(
                np.zeros(total_positions, dtype=np.uint8)
            )

            mismatch_counts = cuda.to_device(
                np.zeros(total_positions, dtype=np.int32)
            )

            threads_per_block = 128
            blocks_per_grid = (
                total_positions + threads_per_block - 1
            ) // threads_per_block

            _gpu_alignment_kernel[blocks_per_grid, threads_per_block](  # type: ignore[reportIndexIssue]
                device_genome,
                device_target,
                max_mismatches,
                match_flags,
                mismatch_counts
            )  # type: ignore[reportIndexIssue]

            flags = match_flags.copy_to_host()
            counts = mismatch_counts.copy_to_host()

            positions = np.flatnonzero(flags).tolist()

            distances = [
                int(counts[position])
                for position in positions
            ]

        else:
            positions, distances = _cpu_alignment_kernel(
                encoded_genome,
                encoded_target,
                max_mismatches
            )

        for position, distance in zip(positions, distances):
            candidate = sequence[position:position + target_length]

            mismatch_positions = [
                index
                for index, (a, b) in enumerate(zip(target, candidate))
                if a != b
            ]

            match = {
                "record_id": record["id"],
                "position": int(position),
                "matched_sequence": candidate,
                "mismatches": int(distance),
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
