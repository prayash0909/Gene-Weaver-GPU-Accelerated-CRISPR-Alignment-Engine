
from dask.base import compute
from dask.delayed import delayed


from .cpu_aligner import find_exact_matches


def find_exact_matches_parallel(
    target_sequence,
    genome_records,
    num_workers=None
):
    """
    Find exact DNA matches using Dask parallel processing.

    Each genome record is processed as an independent task.
    Results preserve the original genome record order.
    """

    if (
        num_workers is not None
        and (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
        )
    ):
        raise TypeError("Number of workers must be an integer.")

    if num_workers is not None and num_workers <= 0:
        raise ValueError("Number of workers must be positive.")

    # Validate target sequence and genome records.
    # Validate target sequence.
    find_exact_matches(target_sequence, [])

# Validate genome records.
    if not isinstance(genome_records, list):
        raise TypeError("Genome records must be a list.")

    if not genome_records:
       return []

    # Create one independent task for each genome record.
    tasks = [
        delayed(find_exact_matches)(
            target_sequence,
            [record]
        )
        for record in genome_records
    ]

    # Execute tasks using Dask's threaded scheduler.
    results = compute(
        *tasks,
        scheduler="threads",
        num_workers=num_workers
    )

    # Combine results in the original record order.
    matches = []

    for record_matches in results:
        matches.extend(record_matches)

    return matches
