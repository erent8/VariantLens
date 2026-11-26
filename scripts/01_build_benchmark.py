"""Build the labelled missense benchmark: ClinVar (P/LP vs B/LB) x AlphaMissense.

Join is done on GRCh38 genomic coordinates (chrom, pos, ref, alt), which gives
each ClinVar SNV the UniProt accession + protein change that AlphaMissense uses
(canonical isoform) -- no fragile HGVS parsing needed.

Temporal hold-out: variants whose labelled classification did NOT exist in the
August-2023 ClinVar release (i.e. published after AlphaMissense and long after
ESM-2's 2021 training data) form a leakage-free "post-2023" test split.

Outputs data/processed/benchmark.parquet
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

PATH = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BEN = {"Benign", "Likely benign", "Benign/Likely benign"}
STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
}
COLS = ["Type", "GeneSymbol", "ClinicalSignificance", "Assembly", "Chromosome",
        "ReviewStatus", "VariationID", "PositionVCF", "ReferenceAlleleVCF",
        "AlternateAlleleVCF", "Name"]


def load_clinvar(path: Path, min_stars: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=COLS, dtype=str, low_memory=False)
    df = df[(df.Assembly == "GRCh38") & (df.Type == "single nucleotide variant")]
    df = df[df.ClinicalSignificance.isin(PATH | BEN)]
    df["stars"] = df.ReviewStatus.map(STARS).fillna(0).astype(int)
    df = df[df.stars >= min_stars]
    df["label"] = df.ClinicalSignificance.isin(PATH).astype(int)
    df = df.rename(columns={"Chromosome": "chrom", "PositionVCF": "pos",
                            "ReferenceAlleleVCF": "ref", "AlternateAlleleVCF": "alt",
                            "GeneSymbol": "gene"})
    df["chrom"] = "chr" + df.chrom
    df["pos"] = pd.to_numeric(df.pos, errors="coerce")
    df = df.dropna(subset=["pos"]).astype({"pos": "int64"})
    # a genomic change must have one consistent label
    key = ["chrom", "pos", "ref", "alt"]
    n_lab = df.groupby(key).label.transform("nunique")
    df = df[n_lab == 1].sort_values("stars", ascending=False).drop_duplicates(key)
    return df[key + ["gene", "label", "stars", "ClinicalSignificance", "VariationID", "Name"]]


def am_header_skip(path: Path) -> int:
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if line.startswith("#CHROM"):
                return i
    raise ValueError("AlphaMissense header not found")


def main():
    cv = load_clinvar(RAW / "variant_summary.txt.gz")
    print(f"ClinVar labelled SNVs (>=1 star): {len(cv):,}")

    old = load_clinvar(RAW / "variant_summary_2023-08.txt.gz", min_stars=0)
    old_keys = set(zip(old.chrom, old.pos, old.ref, old.alt))
    cv["in_clinvar_2023_08"] = [k in old_keys for k in zip(cv.chrom, cv.pos, cv.ref, cv.alt)]
    print(f"  of which labelled already in 2023-08: {cv.in_clinvar_2023_08.sum():,}")

    am_path = RAW / "AlphaMissense_hg38.tsv.gz"
    skip = am_header_skip(am_path)
    parts = []
    for chunk in pd.read_csv(am_path, sep="\t", skiprows=skip, chunksize=5_000_000,
                             dtype={"#CHROM": str, "POS": "int64"}):
        chunk = chunk.rename(columns={"#CHROM": "chrom", "POS": "pos", "REF": "ref", "ALT": "alt"})
        m = chunk.merge(cv, on=["chrom", "pos", "ref", "alt"], how="inner")
        parts.append(m)
        print(f"  AM chunk -> {len(m):,} matches", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["chrom", "pos", "ref", "alt"])
    df = df.rename(columns={"protein_variant": "variant", "uniprot_id": "uniprot"})
    df["split"] = df.in_clinvar_2023_08.map({True: "pre2023", False: "post2023"})
    df.to_parquet(OUT / "benchmark.parquet", index=False)
    print(f"Benchmark: {len(df):,} missense variants, {df.uniprot.nunique():,} proteins")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
