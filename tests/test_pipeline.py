
from typing import cast

import pytest

from geneweaver import pipeline
from geneweaver.pipeline import run_pipeline


def create_fasta(tmp_path):
    fasta_file = tmp_path / "test_genome.fasta"

    fasta_file.write_text(
        ">gene_001 Target with PAM\n"
        "ATGCGTACGTAGCTAGCTAGAGG\n"
        ">gene_002 One mismatch\n"
        "ATGCGTACGTAGATAGCTAGAGG\n"
        ">gene_003 Unrelated sequence\n"
        "GGGAAACCCGGGTTTAAACC\n",
        encoding="utf-8",
    )

    return fasta_file


def test_pipeline_runs_complete_workflow(tmp_path):
    fasta_file = create_fasta(tmp_path)

    result = run_pipeline(
        fasta_file,
        "ATGCGTACGTAGCTAGCTAG",
        max_mismatches=2,
    )

    assert result["genome_records"] == 3
    assert result["actual_backend"] == "CPU"
    assert result["candidates_found"] >= 2
    assert result["candidates"][0]["rank"] == 1


def test_pipeline_ranks_exact_match_first(tmp_path):
    fasta_file = create_fasta(tmp_path)

    result = run_pipeline(
        fasta_file,
        "ATGCGTACGTAGCTAGCTAG",
        max_mismatches=2,
    )

    candidates = result["candidates"]

    assert candidates[0]["mismatches"] == 0
    assert candidates[0]["rank"] == 1


def test_pipeline_detects_pam_sites(tmp_path):
    fasta_file = create_fasta(tmp_path)

    result = run_pipeline(
        fasta_file,
        "ATGCGTACGTAGCTAGCTAG",
    )

    exact_match = next(
        candidate
        for candidate in result["candidates"]
        if candidate["record_id"] == "gene_001"
    )

    assert exact_match["pam_found"] is True
    assert exact_match["pams"][0]["pam_sequence"] == "AGG"


def test_pipeline_falls_back_to_cpu_when_cuda_unavailable(
    tmp_path,
    monkeypatch,
):
    fasta_file = create_fasta(tmp_path)

    monkeypatch.setattr(
        pipeline.cuda,
        "is_available",
        lambda: False,
    )

    result = run_pipeline(
        fasta_file,
        "ATGCGTACGTAGCTAGCTAG",
        use_gpu=True,
    )

    assert result["requested_backend"] == "CUDA"
    assert result["actual_backend"] == "CPU"
    assert result["fallback_reason"] is not None


def test_pipeline_rejects_invalid_target(tmp_path):
    fasta_file = create_fasta(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(fasta_file, "ATGX")


def test_pipeline_rejects_invalid_gpu_option(tmp_path):
    fasta_file = create_fasta(tmp_path)

    with pytest.raises(TypeError):
        run_pipeline(
            fasta_file,
            "ATGC",
            use_gpu=cast(bool, "yes"),
        )


def test_pipeline_rejects_missing_fasta():
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            "missing_genome.fasta",
            "ATGC",
        )
