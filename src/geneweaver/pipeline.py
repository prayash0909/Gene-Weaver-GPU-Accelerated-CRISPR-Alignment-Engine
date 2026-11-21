
from pathlib import Path

from numba import cuda

from .parser import parse_fasta, VALID_DNA_BASES
from .gpu_aligner import find_approximate_matches_accelerated
from .scoring import rank_crispr_candidates


def run_pipeline(
    fasta_path,
    target_sequence,
    max_mismatches=2,
    use_gpu=False,
):
    """
    Run the complete GeneWeaver CRISPR alignment pipeline.

    Steps:
    1. Validate pipeline inputs.
    2. Parse FASTA genome.
    3. Select CUDA or CPU execution.
    4. Find approximate DNA matches.
    5. Detect adjacent PAM sites.
    6. Rank potential CRISPR candidates.

    Automatically falls back to CPU if CUDA is unavailable.
    """

    # Step 1: Validate input parameters.
    if not isinstance(use_gpu, bool):
        raise TypeError("use_gpu must be a boolean.")

    if not isinstance(target_sequence, str):
        raise TypeError("Target sequence must be a string.")

    if not target_sequence:
        raise ValueError("Target sequence cannot be empty.")

    if target_sequence != target_sequence.strip():
        raise ValueError("Target sequence cannot contain surrounding whitespace.")

    target_sequence = target_sequence.upper()

    invalid_bases = set(target_sequence) - VALID_DNA_BASES

    if invalid_bases:
        invalid = "".join(sorted(invalid_bases))
        raise ValueError(f"Target contains invalid DNA bases: {invalid}")

    if (
        isinstance(max_mismatches, bool)
        or not isinstance(max_mismatches, int)
    ):
        raise TypeError("max_mismatches must be a non-negative integer.")

    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative.")

    # Step 2: Parse the FASTA genome.
    fasta_path = Path(fasta_path)
    records = parse_fasta(fasta_path)

    total_bases = sum(record["length"] for record in records)

    # Step 3: Select execution backend.
    requested_backend = "CUDA" if use_gpu else "CPU"
    actual_use_gpu = False
    fallback_reason = None

    if use_gpu:
        if cuda.is_available():
            actual_use_gpu = True
        else:
            fallback_reason = (
                "CUDA is unavailable. CPU fallback was activated."
            )

    actual_backend = "CUDA" if actual_use_gpu else "CPU"

    # Step 4: Perform approximate DNA alignment.
    matches = find_approximate_matches_accelerated(
        target_sequence,
        records,
        max_mismatches=max_mismatches,
        include_pam=True,
        use_gpu=actual_use_gpu,
    )

    # Step 5: Rank CRISPR candidates.
    ranked_candidates = rank_crispr_candidates(matches)

    # Step 6: Return structured pipeline report.
    return {
        "input_file": str(fasta_path),
        "target_sequence": target_sequence,
        "genome_records": len(records),
        "total_bases": total_bases,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "fallback_reason": fallback_reason,
        "candidates_found": len(ranked_candidates),
        "candidates": ranked_candidates,
    }
