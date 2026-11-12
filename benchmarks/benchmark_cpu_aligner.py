import sys
from pathlib import Path
from time import perf_counter
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geneweaver.cpu_aligner import find_exact_matches

GENOME_SIZE = 2_000_000
TARGET = "CGTACGATCGTAGCTAGCTA"
MATCH_POSITIONS = [100_000, 900_000, 1_700_000]
RUNS = 5


def main():
    genome = list("A" * GENOME_SIZE)

    for position in MATCH_POSITIONS:
        genome[position:position + len(TARGET)] = TARGET

    genome_sequence = "".join(genome)
    records = [{"id": "benchmark_genome", "sequence": genome_sequence}]

    execution_times = []
    matches = []

    for _ in range(RUNS):
        start = perf_counter()
        matches = find_exact_matches(TARGET, records)
        execution_times.append(perf_counter() - start)

    average_time = median(execution_times)
    throughput = GENOME_SIZE / average_time

    print("\n===== GeneWeaver CPU Benchmark =====")
    print(f"Genome size       : {GENOME_SIZE:,} bases")
    print(f"Target length     : {len(TARGET)} bases")
    print(f"Expected matches  : {len(MATCH_POSITIONS)}")
    print(f"Actual matches    : {len(matches)}")
    print(f"Median time       : {average_time:.6f} seconds")
    print(f"Throughput        : {throughput:,.0f} bases/second")
    print(f"Match positions   : {[m['position'] for m in matches]}")

    assert [m["position"] for m in matches] == MATCH_POSITIONS
    print("\nBenchmark validation: PASSED")


if __name__ == "__main__":
    main()
