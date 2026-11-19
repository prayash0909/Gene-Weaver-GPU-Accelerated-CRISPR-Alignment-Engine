# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a Bioinformatics and High-Performance Computing project for identifying potential CRISPR off-target DNA sequences.

## Features

- FASTA parsing and DNA validation using BioPython
- DNA sequence chunking
- Python CPU alignment baseline
- Numba-accelerated approximate matching
- Optional NVIDIA CUDA alignment
- CPU fallback when CUDA is unavailable
- Dask-based parallel genome scanning
- SpCas9 PAM detection
- CRISPR candidate scoring and ranking
- End-to-end analysis pipeline
- Automated tests and performance benchmarks

## Technology Stack

- Python 3.11
- BioPython
- NumPy
- Numba
- Dask
- Textual
- Pytest

## Running the Project

Run the pipeline:

```bash
python run_pipeline.py
```

Run tests:

```bash
python -m pytest -q
```

Run the benchmark:

```bash
python benchmarks/benchmark_pipeline.py
```

## Benchmark Results

Local synthetic CPU benchmark:

- Genome records: 8
- Total DNA bases: 160,368
- Benchmark runs: 7
- Candidates detected: 16
- Result verification: Passed

Execution time varies depending on system load and hardware.

## Hardware Note

Local development uses Intel Iris Xe Graphics. NVIDIA CUDA execution is unavailable on this machine. The optional CUDA path has not been validated locally. CPU execution is used for local testing.

## Project Status

Core FASTA parsing, CPU alignment, approximate matching, PAM detection, candidate ranking, parallel scanning, and end-to-end pipeline integration have been implemented and tested.

## Internship Project

Infotact Solutions | Python Development Internship

Project Duration: 10 November 2026 - 10 December 2026
