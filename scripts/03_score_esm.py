"""Score benchmark variants with an ESM-2 model.

  python scripts/03_score_esm.py --model esm2_650M                # wild-type marginal, all variants
  python scripts/03_score_esm.py --model esm2_650M --mode mm --sample 25000

* wt: one forward pass per protein -> full L x 20 landscape saved to
      data/processed/landscapes/<model>/<uniprot>.npy (float16) and the LLR of
      every labelled variant -> results/scores_<model>.parquet   (column <model>)
* mm: masked-marginal LLR for a random subset of variants (one pass per
      position; ~40x more compute) -> results/scores_<model>_mm.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from variantlens.esm_scoring import AA_IDX, ESMScorer  # noqa: E402

PROC, RES = ROOT / "data/processed", ROOT / "results"
KEY = ["chrom", "pos", "ref", "alt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_650M")
    ap.add_argument("--mode", choices=["wt", "mm"], default="wt")
    ap.add_argument("--sample", type=int, default=25_000)
    args = ap.parse_args()

    df = pd.read_parquet(PROC / "benchmark.parquet")
    if args.mode == "mm":
        df = df.sample(min(args.sample, len(df)), random_state=0).reset_index(drop=True)
    seqs = pd.read_parquet(PROC / "sequences.parquet").set_index("uniprot").sequence.to_dict()
    scorer = ESMScorer(args.model)
    print(f"{args.model} [{args.mode}] on {scorer.device}: {len(df):,} variants", flush=True)

    land_dir = PROC / "landscapes" / args.model
    land_dir.mkdir(parents=True, exist_ok=True)
    out = np.full(len(df), np.nan, np.float32)
    groups = df.groupby("uniprot").indices
    order = sorted(groups, key=lambda u: -len(seqs[u]))  # longest first: OOM shows up early
    t0 = time.time()
    for n, u in enumerate(order, 1):
        idx, seq = groups[u], seqs[u]
        var = df.variant.values[idx]
        pos = np.array([int(v[1:-1]) - 1 for v in var])
        mt = np.array([AA_IDX[v[-1]] for v in var])
        if args.mode == "wt":
            f = land_dir / f"{u}.npy"
            if f.exists():
                land = np.load(f).astype(np.float32)
            else:
                land = scorer.wt_marginal(seq)
                if args.model == "esm2_650M":  # only the demo model's landscapes are kept
                    np.save(f, land.astype(np.float16))
            out[idx] = land[pos, mt]
        else:
            try:
                res = scorer.masked_marginal(seq, pos)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                scorer.token_budget //= 4
                res = scorer.masked_marginal(seq, pos)
                scorer.token_budget *= 4
            out[idx] = [res[p][m] for p, m in zip(pos, mt)]
        if n % 1000 == 0 or n == len(order):
            print(f"  {n:,}/{len(order):,} proteins  {(time.time() - t0) / 60:.1f} min", flush=True)

    res = df[KEY].copy()
    res[args.model] = out
    RES.mkdir(exist_ok=True)
    suffix = "" if args.mode == "wt" else "_mm"
    res.to_parquet(RES / f"scores_{args.model}{suffix}.parquet", index=False)
    print(f"done: {np.isnan(out).sum()} missing, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
