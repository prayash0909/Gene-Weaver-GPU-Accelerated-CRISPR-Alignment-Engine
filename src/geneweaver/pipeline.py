
from pathlib import Path

from numba import cuda

from .parser import parse_fasta
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
    1. Parse FASTA genome.
    2. Select CUDA or CPU execution.
    3. Find approximate DNA matches.
    4. Detect adjacent PAM sites.
    5. Rank potential CRISPR off-target candidates.

    Automatically falls back to CPU if CUDA is unavailable.
    """

    if not isinstance(use_gpu, bool):
        raise TypeError("use_gpu must be a boolean.")

    if not isinstance(target_sequence, str):
        raise TypeError("Target sequence must be a string.")

    if not target_sequence:
        raise ValueError("Target sequence cannot be empty.")

    # Step 1: Parse the FASTA genome.
    records = parse_fasta(fasta_path)

    total_bases = sum(record["length"] for record in records)

    # Step 2: Select the execution backend.
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

    # Step 3: Perform approximate DNA alignment.
    matches = find_approximate_matches_accelerated(
        target_sequence,
        records,
        max_mismatches=max_mismatches,
        include_pam=True,
        use_gpu=actual_use_gpu,
    )

    # Step 4: Rank CRISPR candidates.
    ranked_candidates = rank_crispr_candidates(matches)

    # Step 5: Return a structured pipeline report.
    return {
        "input_file": str(Path(fasta_path)),
        "target_sequence": target_sequence.upper(),
        "genome_records": len(records),
        "total_bases": total_bases,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "fallback_reason": fallback_reason,
        "candidates_found": len(ranked_candidates),
        "candidates": ranked_candidates,
    }
