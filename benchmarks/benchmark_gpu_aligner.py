
import statistics
import time

from geneweaver.gpu_aligner import find_approximate_matches_accelerated
from geneweaver.scoring import find_approximate_matches


def run_benchmark():
    target = "ATGCGTACGTAGCTAGCTAG"

    genome_records = [
        {
            "id": f"gene_{index:03d}",
            "sequence": (
                "ATGCGTACGTAGCTAGCTAG"
                + "A" * 10000
                + "ATGCGTACGTAGATAGCTAG"
                + "C" * 10000
            ),
        }
        for index in range(8)
    ]

    runs = 7

    # Warm up Numba compilation before measuring.
    find_approximate_matches_accelerated(
        target,
        genome_records,
        max_mismatches=2,
    )

    baseline_times = []
    accelerated_times = []

    baseline_results = None
    accelerated_results = None

    for _ in range(runs):
        start = time.perf_counter()
        baseline_results = find_approximate_matches(
            target,
            genome_records,
            max_mismatches=2,
        )
        baseline_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        accelerated_results = find_approximate_matches_accelerated(
            target,
            genome_records,
            max_mismatches=2,
        )
        accelerated_times.append(time.perf_counter() - start)

    if baseline_results is None or accelerated_results is None:
        raise RuntimeError("Benchmark did not produce any results.")

    print("\nGeneWeaver CPU Alignment Benchmark")
    print("----------------------------------")
    print(f"Genome records: {len(genome_records)}")
    print(f"Total bases: {sum(len(r['sequence']) for r in genome_records):,}")
    print(f"Target length: {len(target)}")
    print(f"Benchmark runs: {runs}")

    baseline_median = statistics.median(baseline_times)
    accelerated_median = statistics.median(accelerated_times)

    print(f"\nPython scoring median: {baseline_median:.6f} seconds")
    print(f"Numba CPU median:      {accelerated_median:.6f} seconds")

    if accelerated_median > 0:
        speedup = baseline_median / accelerated_median
        print(f"Measured speedup:      {speedup:.2f}x")

    print(f"\nPython matches: {len(baseline_results)}")
    print(f"Numba matches:  {len(accelerated_results)}")

    identical = baseline_results == accelerated_results
    print(f"Results identical: {identical}")

    assert identical, "Benchmark implementations returned different results."

    print("\nBenchmark verification: PASSED")


if __name__ == "__main__":
    run_benchmark()
