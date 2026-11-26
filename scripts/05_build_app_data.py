"""Package everything the web demo needs into app/data (self-contained, deployable).

* variants.parquet      labelled ClinVar variants with ESM + AM scores
* proteins.parquet      uniprot, gene, sequence, per-gene AUROCs
* am_landscape.parquet  all SNV-reachable AlphaMissense scores for the proteins
* landscapes/<acc>.npy  (optional copy) ESM-2 650M wild-type-marginal L x 20 matrices
* calibration.json      logistic map ESM LLR -> P(pathogenic), thresholds
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROC, RES, APP = ROOT / "data/processed", ROOT / "results", ROOT / "app/data"
MODEL = "esm2_650M"


def main():
    APP.mkdir(parents=True, exist_ok=True)
    bench = pd.read_parquet(PROC / "benchmark.parquet")
    sc = pd.read_parquet(RES / f"scores_{MODEL}.parquet")
    df = bench.merge(sc, on=["chrom", "pos", "ref", "alt"])
    v = df[["gene", "uniprot", "variant", "label", "ClinicalSignificance", "stars", "split",
            "VariationID", "am_pathogenicity", "am_class", MODEL]].rename(columns={MODEL: "esm"})
    v.to_parquet(APP / "variants.parquet", index=False)

    # calibration: ESM LLR -> probability, threshold at Youden's J
    lr = LogisticRegression().fit(v[["esm"]], v.label)
    fpr, tpr, thr = roc_curve(v.label, -v.esm)
    t = -thr[np.argmax(tpr - fpr)]
    cal = {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
           "esm_threshold": float(t), "am_thresholds": [0.34, 0.564]}
    (APP / "calibration.json").write_text(json.dumps(cal, indent=2))
    print("calibration", cal)

    seqs = pd.read_parquet(PROC / "sequences.parquet")
    genes = v.groupby("uniprot").agg(gene=("gene", "first"), n=("label", "size"), n_path=("label", "sum")).reset_index()
    prot = seqs.merge(genes, on="uniprot")
    ren = {"AlphaMissense": "auroc_am", "ESM-2 650M": "auroc_esm", "VariantLens-Seq": "auroc_vl_seq", "VariantLens+AM": "auroc_vl_am"}
    pg = pd.read_csv(RES / "per_gene_auroc.csv")
    pg = pg[["gene"] + [c for c in ren if c in pg]].rename(columns=ren)
    prot = prot.merge(pg, on="gene", how="left")
    prot.to_parquet(APP / "proteins.parquet", index=False)

    # landscapes are read straight from data/processed/landscapes/<MODEL> (no copy);
    # pass --copy-landscapes to make app/ fully self-contained for deployment
    if "--copy-landscapes" in sys.argv:
        src, dst = PROC / "landscapes" / MODEL, APP / "landscapes"
        dst.mkdir(exist_ok=True)
        for u in prot.uniprot:
            if (src / f"{u}.npy").exists() and not (dst / f"{u}.npy").exists():
                shutil.copy(src / f"{u}.npy", dst / f"{u}.npy")

    # full AlphaMissense (SNV-reachable) scores for these proteins
    from importlib import import_module
    skip = import_module("01_build_benchmark").am_header_skip(ROOT / "data/raw/AlphaMissense_hg38.tsv.gz")
    keep, parts = set(prot.uniprot), []
    for ch in pd.read_csv(ROOT / "data/raw/AlphaMissense_hg38.tsv.gz", sep="\t", skiprows=skip,
                          usecols=["uniprot_id", "protein_variant", "am_pathogenicity"], chunksize=5_000_000):
        parts.append(ch[ch.uniprot_id.isin(keep)])
    am = pd.concat(parts).drop_duplicates(["uniprot_id", "protein_variant"])
    am["am_pathogenicity"] = am.am_pathogenicity.astype("float32")
    # sorted + small row groups -> the app reads one protein via predicate push-down in ms
    am.rename(columns={"uniprot_id": "uniprot", "protein_variant": "variant"}).sort_values("uniprot").to_parquet(
        APP / "am_landscape.parquet", index=False, row_group_size=100_000)
    print(f"proteins {len(prot):,}  AM rows {len(am):,}")


if __name__ == "__main__":
    main()
