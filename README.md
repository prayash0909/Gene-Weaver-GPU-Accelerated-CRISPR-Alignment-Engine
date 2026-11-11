# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a bioinformatics and high-performance computing project designed to identify potential CRISPR off-target DNA sequences in large genomic FASTA files.

## Project Overview

The goal is to build an efficient sequence alignment pipeline that processes genomic data, compares target DNA sequences, and identifies potential off-target matches.

## Key Features

- FASTA genome parsing using BioPython
- DNA sequence cleaning and chunking
- Pure Python CPU alignment baseline
- NumPy-based sequence processing
- Numba-based computational acceleration
- Dask-based distributed processing
- Textual terminal dashboard
- Off-target scoring and ranking

## Technology Stack

- Python 3.11
- BioPython
- NumPy
- Numba
- Dask
- Textual
- Pytest

## Project Structure

```text
GeneWeaver/
├── data/
│   ├── input/
│   └── samples/
├── src/
│   └── geneweaver/
│       ├── parser.py
│       ├── cpu_aligner.py
│       ├── gpu_aligner.py
│       ├── scoring.py
│       └── scheduler.py
├── tests/
├── benchmarks/
├── tui/
├── docs/
├── requirements.txt
└── README.md
```

## Hardware Note

Local development currently uses an Intel Iris Xe Graphics system. NVIDIA CUDA execution is not available on this machine. GPU acceleration will be evaluated on a compatible NVIDIA CUDA environment. CPU execution will serve as the baseline.

## Project Status

Currently in the initial setup and CPU baseline development phase.

## Internship Project

Infotact Solutions | Python Development Internship  
Project Duration: 10 November 2026 – 10 December 2026
