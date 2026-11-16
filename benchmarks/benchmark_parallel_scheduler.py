
import sys
from pathlib import Path
from time import perf_counter
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geneweaver.cpu_aligner import find_exact_matches
from src.geneweaver.scheduler import find_exact_matches_parallel


RECORD_COUNT = 8
RECORD_SIZE = 250_000
TARGET = "CGTACGATCGTAGCTAGCTA"
MATCH_POSITIONS = [10_000, 90_000, 170_000]
RUNS = 7


def create_genome_records():
    """Create multiple independent synthetic genome records."""

    records = []

    for index in range(RECORD_COUNT):
        genome = list("A" * RECORD_SIZE)

        for position in MATCH_POSITIONS:
            genome[position:position + len(TARGET)] = TARGET

        records.append({
            "id": f"genome_{index + 1:03d}",
            "sequence": "".join(genome),
        })

    return records


def main():
    records = create_genome_records()

    total_genome_size = RECORD_COUNT * RECORD_SIZE

    expected_matches = [
        {
            "record_id": record["id"],
            "position": position,
            "matched_sequence": TARGET,
        }
        for record in records
        for position in MATCH_POSITIONS
    ]

    # Warm-up both execution paths.
    find_exact_matches(TARGET, records)
    find_exact_matches_parallel(TARGET, records)

    sequential_times = []
    parallel_times = []

    sequential_matches = []
    parallel_matches = []

    for _ in range(RUNS):
        start = perf_counter()
        sequential_matches = find_exact_matches(TARGET, records)
        sequential_times.append(perf_counter() - start)

        start = perf_counter()
        parallel_matches = find_exact_matches_parallel(TARGET, records)
        parallel_times.append(perf_counter() - start)

    sequential_median = median(sequential_times)
    parallel_median = median(parallel_times)

    speedup = sequential_median / parallel_median

    print("\n===== GeneWeaver Dask Scaling Benchmark =====")
    print(f"Genome records       : {RECORD_COUNT}")
    print(f"Record size          : {RECORD_SIZE:,} bases")
    print(f"Total genome size    : {total_genome_size:,} bases")
    print(f"Target length        : {len(TARGET)} bases")
    print(f"Runs                 : {RUNS}")
    print(f"Expected matches     : {len(expected_matches)}")

    print(f"\nSequential median    : {sequential_median:.6f} seconds")
    print(f"Dask parallel median : {parallel_median:.6f} seconds")
    print(f"Measured speedup     : {speedup:.2f}x")

    print(f"\nSequential matches   : {len(sequential_matches)}")
    print(f"Parallel matches     : {len(parallel_matches)}")

    assert sequential_matches == expected_matches
    assert parallel_matches == expected_matches

    print("\nResult verification  : PASSED")


if __name__ == "__main__":
    main()
