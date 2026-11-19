
import sys
import statistics
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geneweaver.pipeline import run_pipeline

TARGET = "ATGCGTACGTAGCTAGCTAG"
RECORD_COUNT = 8
RECORD_SIZE = 20_000
RUNS = 7


def create_benchmark_fasta(path):
    """Create a synthetic FASTA genome for pipeline benchmarking."""

    with path.open("w", encoding="ascii") as fasta:
        for index in range(RECORD_COUNT):
            sequence = (
                "A" * 5000
                + TARGET
                + "AGG"
                + "C" * 5000
                + "A" * 5000
                + "ATGCGTACGTAGATAGCTAG"
                + "AGG"
                + "C" * 5000
            )

            fasta.write(f">benchmark_gene_{index + 1:03d}\n")
            fasta.write(sequence + "\n")


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        fasta_path = Path(temporary_directory) / "benchmark.fasta"

        create_benchmark_fasta(fasta_path)

        # Warm-up Numba before measuring execution time.
        warmup = run_pipeline(
            fasta_path,
            TARGET,
            max_mismatches=2,
            use_gpu=False,
        )

        execution_times = []
        final_report = None

        for _ in range(RUNS):
            start = time.perf_counter()

            final_report = run_pipeline(
                fasta_path,
                TARGET,
                max_mismatches=2,
                use_gpu=False,
            )

            execution_times.append(time.perf_counter() - start)

        if final_report is None:
            raise RuntimeError("Benchmark did not produce a result.")

        median_time = statistics.median(execution_times)

        print("\n===== GeneWeaver End-to-End Pipeline Benchmark =====")
        print(f"Genome records       : {final_report['genome_records']}")
        print(f"Total genome bases   : {final_report['total_bases']:,}")
        print(f"Target length        : {len(TARGET)}")
        print(f"Execution backend    : {final_report['actual_backend']}")
        print(f"Benchmark runs       : {RUNS}")

        print(f"\nWarm-up candidates   : {warmup['candidates_found']}")
        print(f"Final candidates     : {final_report['candidates_found']}")
        print(f"Median execution time: {median_time:.6f} seconds")

        print("\nCandidate ranking:")
        for candidate in final_report["candidates"]:
            print(
                f"{candidate['record_id']} | "
                f"Position: {candidate['position']} | "
                f"Mismatches: {candidate['mismatches']} | "
                f"PAM: {candidate['pam_found']} | "
                f"Rank: {candidate['rank']}"
            )

        # Verify deterministic results across repeated runs.
        assert warmup["candidates"] == final_report["candidates"]
        assert final_report["actual_backend"] == "CPU"

        print("\nResult verification  : PASSED")
        print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
