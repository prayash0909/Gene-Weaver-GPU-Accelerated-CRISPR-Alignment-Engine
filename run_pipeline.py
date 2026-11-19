
from pathlib import Path
import sys

# Make the src package available when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geneweaver.pipeline import run_pipeline


def main():
    fasta_file = PROJECT_ROOT / "data" / "samples" / "sample_genome.fasta"

    target = "ATGCGTACGTAGCTAGCTAG"

    report = run_pipeline(
        fasta_path=fasta_file,
        target_sequence=target,
        max_mismatches=2,
        use_gpu=False,
    )

    print("\n========== GENEWEAVER PIPELINE REPORT ==========")
    print(f"Input file       : {report['input_file']}")
    print(f"Genome records   : {report['genome_records']}")
    print(f"Total DNA bases  : {report['total_bases']}")
    print(f"Requested backend: {report['requested_backend']}")
    print(f"Actual backend   : {report['actual_backend']}")
    print(f"Candidates found : {report['candidates_found']}")

    if report["fallback_reason"]:
        print(f"Fallback reason  : {report['fallback_reason']}")

    print("\n---------- RANKED CRISPR CANDIDATES ----------")

    for candidate in report["candidates"]:
        print(
            f"\nRank: {candidate['rank']}"
            f"\nGene: {candidate['record_id']}"
            f"\nPosition: {candidate['position']}"
            f"\nSequence: {candidate['matched_sequence']}"
            f"\nMismatches: {candidate['mismatches']}"
            f"\nPAM found: {candidate['pam_found']}"
            f"\nPAM details: {candidate['pams']}"
        )

    print("\n========== PIPELINE COMPLETED ==========\n")


if __name__ == "__main__":
    main()
