
import random
import statistics
import time

from src.geneweaver.scoring import _hamming_distance_with_limit


DNA_BASES = "ACGT"


def count_matches_baseline(target, genome, max_mismatches):
    """Compare every base, even after exceeding the mismatch limit."""

    count = 0
    target_length = len(target)

    for position in range(len(genome) - target_length + 1):
        candidate = genome[position:position + target_length]

        mismatches = sum(
            a != b for a, b in zip(target, candidate)
        )

        if mismatches <= max_mismatches:
            count += 1

    return count


def count_matches_optimized(target, genome, max_mismatches):
    """Stop comparing as soon as the mismatch limit is exceeded."""

    count = 0
    target_length = len(target)

    for position in range(len(genome) - target_length + 1):
        candidate = genome[position:position + target_length]

        distance = _hamming_distance_with_limit(
            target,
            candidate,
            max_mismatches
        )

        if distance is not None:
            count += 1

    return count


def benchmark(function, target, genome, max_mismatches, repeats=5):
    durations = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()

        result = function(target, genome, max_mismatches)

        durations.append(time.perf_counter() - start)

    return result, statistics.median(durations)


def main():
    rng = random.Random(42)

    genome = list(
        "".join(rng.choices(DNA_BASES, k=250_000))
    )

    target = "".join(rng.choices(DNA_BASES, k=20))

    # Insert two known approximate matches.
    genome[50_000:50_020] = list(target)
    genome[150_000:150_020] = list(target)

    # Introduce one mismatch in the second match.
    original_base = genome[150_007]
    genome[150_007] = next(
        base for base in DNA_BASES if base != original_base
    )

    genome = "".join(genome)
    max_mismatches = 2

    baseline_count, baseline_time = benchmark(
        count_matches_baseline,
        target,
        genome,
        max_mismatches
    )

    optimized_count, optimized_time = benchmark(
        count_matches_optimized,
        target,
        genome,
        max_mismatches
    )

    if baseline_count != optimized_count:
        raise RuntimeError("Benchmark results do not match!")

    speedup = baseline_time / optimized_time

    print("APPROXIMATE DNA MATCHING BENCHMARK")
    print("---------------------------------")
    print(f"Genome length: {len(genome):,} bases")
    print(f"Target length: {len(target)} bases")
    print(f"Maximum mismatches: {max_mismatches}")
    print(f"Matches detected: {optimized_count}")
    print(f"Baseline median: {baseline_time:.6f} seconds")
    print(f"Optimized median: {optimized_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print("Result verification: PASSED")


if __name__ == "__main__":
    main()
