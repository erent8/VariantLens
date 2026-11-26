"""Train the VariantLens meta-predictors on ESM-2 landscape features.

* VariantLens-Seq : sequence-only (ESM-2 landscape features + amino-acid chemistry)
                    -> works for *any* protein, including designed ones.
* VariantLens+AM  : the same features plus the AlphaMissense score.

Evaluation uses out-of-fold predictions from 5-fold cross-validation grouped by
GENE, so a model is never tested on a gene it was trained on (no within-gene
label leakage). Final models are refit on all data for the web app.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from variantlens.features import CATEGORICAL, FEATURES, landscape_features  # noqa: E402

PROC, RES = ROOT / "data/processed", ROOT / "results"
KEY = ["chrom", "pos", "ref", "alt"]
LAND = PROC / "landscapes" / "esm2_650M"
# Only features computable from the 650M landscape are used, so the trained model
# can score ANY substitution of ANY sequence in the web app (not just ClinVar ones).
SMALL: list[str] = []


def build_features() -> pd.DataFrame:
    f = PROC / "features.parquet"
    if f.exists():
        return add_small(pd.read_parquet(f))
    df = pd.read_parquet(PROC / "benchmark.parquet")
    seqs = pd.read_parquet(PROC / "sequences.parquet").set_index("uniprot").sequence.to_dict()
    parts, t0 = [], time.time()
    for n, (u, g) in enumerate(df.groupby("uniprot"), 1):
        L = np.load(LAND / f"{u}.npy").astype(np.float32)
        x = landscape_features(L, seqs[u], g.variant.tolist())
        x.index = g.index
        parts.append(x)
        if n % 3000 == 0:
            print(f"  features {n:,} proteins {time.time() - t0:.0f}s", flush=True)
    feats = pd.concat(parts).loc[df.index]
    out = pd.concat([df[KEY + ["gene", "label", "am_pathogenicity", "split"]], feats], axis=1)
    out.to_parquet(f, index=False)
    return add_small(out)


def add_small(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-shot LLRs of the smaller ESM-2 models: a cheap model ensemble."""
    for m in SMALL:
        sf = RES / f"scores_{m}.parquet"
        if sf.exists():
            df = df.merge(pd.read_parquet(sf).rename(columns={m: f"llr_{m}"}), on=KEY, how="left")
    return df


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, categorical_features=CATEGORICAL,
                                          random_state=0)


def main():
    df = build_features()
    small = [c for c in df if c.startswith("llr_")]
    sets = {"vl_seq": FEATURES + small, "vl_am": FEATURES + small + ["am_pathogenicity"]}
    y, groups = df.label.values, df.gene.values
    out = df[KEY].copy()
    importance, rng = {}, np.random.default_rng(0)
    for name, cols in sets.items():
        X = df[cols]
        oof = np.zeros(len(df))
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
            clf = model().fit(X.iloc[tr], y[tr])
            oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
            if k == 0:  # permutation importance on held-out genes of the first fold
                sub = rng.choice(te, min(30_000, len(te)), replace=False)
                pi = permutation_importance(clf, X.iloc[sub], y[sub], scoring="roc_auc", n_repeats=3, random_state=0)
                importance[name] = pd.Series(pi.importances_mean, index=cols)
        out[name] = oof
        post = (df.split == "post2023").values
        print(f"{name:7s}  OOF AUROC all {roc_auc_score(y, oof):.4f}   post2023 {roc_auc_score(y[post], oof[post]):.4f}", flush=True)
        final = model().fit(X, y)
        with open(RES / f"model_{name}.pkl", "wb") as fh:
            pickle.dump({"model": final, "features": cols}, fh)
    out.to_parquet(RES / "scores_variantlens.parquet", index=False)
    plot_importance(importance)


def plot_importance(imp: dict[str, pd.Series]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nice = {"llr": "ESM LLR of this substitution", "pos_mean": "position: mean LLR", "pos_min": "position: min LLR",
            "pos_std": "position: LLR spread", "mt_rank": "rank among 19 substitutions", "nbr5_mean": "±5-residue neighbourhood",
            "nbr15_mean": "±15-residue neighbourhood", "rel_pos": "relative position", "log_len": "protein length",
            "d_hydro": "Δ hydropathy", "d_volume": "Δ side-chain volume", "wt_aa": "wild-type residue",
            "mt_aa": "mutant residue", "am_pathogenicity": "AlphaMissense score"}
    fig, axes = plt.subplots(1, len(imp), figsize=(5.2 * len(imp), 4.4))
    for ax, (name, s) in zip(np.atleast_1d(axes), imp.items()):
        s = s.sort_values()
        ax.barh([nice.get(i, i) for i in s.index], s.values, color="#f2a541" if name == "vl_seq" else "#3d405b")
        ax.set(title={"vl_seq": "VariantLens-Seq", "vl_am": "VariantLens+AM"}[name], xlabel="AUROC drop when shuffled")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "feature_importance.png", dpi=150)
    pd.DataFrame(imp).to_csv(RES / "feature_importance.csv")


if __name__ == "__main__":
    main()
