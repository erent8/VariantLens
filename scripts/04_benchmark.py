"""Benchmark ESM-2 (zero-shot) against AlphaMissense on ClinVar missense variants.

Reports
* global AUROC / AUPRC with 95% bootstrap CIs, on all variants and on the
  leakage-free post-2023 split
* per-gene AUROC (genes with >=5 pathogenic and >=5 benign variants) - the
  clinically relevant question: "can the model rank variants *within* a gene?"
* ESM-2 scaling curve (8M -> 650M / 3B parameters)
* masked vs wild-type marginal
* a simple ESM+AlphaMissense ensemble evaluated with gene-grouped CV
* the most discordant variants between the two methods
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data/processed", ROOT / "results", ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
KEY = ["chrom", "pos", "ref", "alt"]
SIZES = {"esm2_8M": 8e6, "esm2_35M": 35e6, "esm2_150M": 150e6, "esm2_650M": 650e6, "esm2_3B": 3e9}
COL = {"AlphaMissense": "#2a6f97", "ESM-2 650M": "#e07a5f", "VariantLens-Seq": "#f2a541", "VariantLens+AM": "#3d405b"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
rng = np.random.default_rng(0)


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROC / "benchmark.parquet")
    models = []
    for m in SIZES:
        f = RES / f"scores_{m}.parquet"
        if f.exists():
            df = df.merge(pd.read_parquet(f), on=KEY, how="left")
            models.append(m)
    # pathogenicity scores: higher = more pathogenic
    for m in models:
        df[f"s_{m}"] = -df[m]
    df["s_AlphaMissense"] = df.am_pathogenicity
    vl = RES / "scores_variantlens.parquet"  # out-of-fold, gene-grouped CV predictions
    if vl.exists():
        df = df.merge(pd.read_parquet(vl), on=KEY, how="left")
    return df.dropna(subset=["s_AlphaMissense"] + [f"s_{m}" for m in models]), models


def boot_ci(y, s, fn, n=1000):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def global_metrics(d, cols):
    y = d.label.values
    out = {}
    for name, c in cols.items():
        s = d[c].values
        out[name] = {"auroc": roc_auc_score(y, s), "auroc_ci": boot_ci(y, s, roc_auc_score),
                     "auprc": average_precision_score(y, s), "auprc_ci": boot_ci(y, s, average_precision_score, 300)}
    out["_n"] = {"variants": int(len(d)), "pathogenic": int(y.sum()), "benign": int((1 - y).sum()),
                 "proteins": int(d.uniprot.nunique())}
    return out


def per_gene(d, cols, min_each=5):
    rows = []
    for g, x in d.groupby("gene"):
        if x.label.sum() >= min_each and (1 - x.label).sum() >= min_each:
            rows.append({"gene": g, "n": len(x), "n_path": int(x.label.sum()),
                         **{n: roc_auc_score(x.label, x[c]) for n, c in cols.items()}})
    return pd.DataFrame(rows)


def main():
    df, models = load()
    best = "esm2_650M" if "esm2_650M" in models else models[-1]
    cols = {"AlphaMissense": "s_AlphaMissense", "ESM-2 650M": f"s_{best}"}
    if "vl_seq" in df:
        cols |= {"VariantLens-Seq": "vl_seq", "VariantLens+AM": "vl_am"}
    post = df[df.split == "post2023"]

    report = {"all": global_metrics(df, cols), "post2023": global_metrics(post, cols)}
    report["scaling"] = {m: {"params": SIZES[m],
                             "auroc_all": roc_auc_score(df.label, df[f"s_{m}"]),
                             "auroc_post2023": roc_auc_score(post.label, post[f"s_{m}"])} for m in models}
    mmf = RES / f"scores_{best}_mm.parquet"
    if mmf.exists():  # ablation on the random subset scored with masked marginals
        mm = pd.read_parquet(mmf).rename(columns={best: "mm"}).merge(df, on=KEY).dropna(subset=["mm"])
        report["masked_vs_wt"] = {"n": int(len(mm)),
                                  "masked_marginal": roc_auc_score(mm.label, -mm.mm),
                                  "wildtype_marginal": roc_auc_score(mm.label, mm[f"s_{best}"]),
                                  "alphamissense": roc_auc_score(mm.label, mm.s_AlphaMissense),
                                  "spearman_mm_wt": float(pd.Series(mm.mm).corr(mm[best], method="spearman"))}

    pg = per_gene(df, cols)
    pg.to_csv(RES / "per_gene_auroc.csv", index=False)
    w = wilcoxon(pg["AlphaMissense"], pg["ESM-2 650M"])
    report["per_gene"] = {"n_genes": len(pg), **{f"mean_{k}": float(pg[k].mean()) for k in cols},
                          **{f"median_{k}": float(pg[k].median()) for k in cols},
                          "esm_better_genes": int((pg["ESM-2 650M"] > pg["AlphaMissense"]).sum()),
                          "wilcoxon_p": float(w.pvalue)}
    if "VariantLens+AM" in pg:
        report["per_gene"] |= {
            "vl_am_better_genes": int((pg["VariantLens+AM"] > pg["AlphaMissense"]).sum()),
            "vl_am_worse_genes": int((pg["VariantLens+AM"] < pg["AlphaMissense"]).sum()),
            "wilcoxon_p_vl_am": float(wilcoxon(pg["AlphaMissense"], pg["VariantLens+AM"]).pvalue),
            "vl_seq_better_than_esm_genes": int((pg["VariantLens-Seq"] > pg["ESM-2 650M"]).sum())}

    # discordant variants: confident disagreement between the two methods
    r_am = rankdata(df.s_AlphaMissense) / len(df)
    r_esm = rankdata(df[f"s_{best}"]) / len(df)
    df["discord"] = r_esm - r_am
    keep = ["gene", "uniprot", "variant", "ClinicalSignificance", "stars", "am_pathogenicity", best, "discord"]
    disc = pd.concat([df.nlargest(25, "discord")[keep], df.nsmallest(25, "discord")[keep]])
    disc.to_csv(RES / "discordant_variants.csv", index=False)
    report["discordance"] = {
        "esm_right_when_disagree": float(((r_esm > r_am) == (df.label == 1))[np.abs(df.discord) > 0.5].mean()),
        "n_strong_disagreements": int((np.abs(df.discord) > 0.5).sum()),
    }
    (RES / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    figures(df, post, pg, report, models, best, cols)


def figures(df, post, pg, report, models, best, cols):
    # 1. ROC curves: all vs post-2023
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, d, title in [(axes[0], df, "All ClinVar (≥1★)"), (axes[1], post, "Post-2023 hold-out")]:
        for n, c in cols.items():
            fpr, tpr, _ = roc_curve(d.label, d[c])
            ax.plot(fpr, tpr, color=COL[n], lw=2, label=f"{n}  AUROC {roc_auc_score(d.label, d[c]):.3f}")
        ax.plot([0, 1], [0, 1], ls=":", c="grey")
        ax.set(title=f"{title}  (n={len(d):,})", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "roc.png"); plt.close(fig)

    # 2. scaling curve
    fig, ax = plt.subplots(figsize=(5.2, 4))
    sc = report["scaling"]
    x = [sc[m]["params"] for m in models]
    ax.plot(x, [sc[m]["auroc_all"] for m in models], "o-", c=COL["ESM-2 650M"], label="ESM-2 (all)")
    ax.plot(x, [sc[m]["auroc_post2023"] for m in models], "s--", c=COL["ESM-2 650M"], alpha=.6, label="ESM-2 (post-2023)")
    ax.axhline(report["all"]["AlphaMissense"]["auroc"], c=COL["AlphaMissense"], ls="-", label="AlphaMissense (all)")
    ax.axhline(report["post2023"]["AlphaMissense"]["auroc"], c=COL["AlphaMissense"], ls="--", alpha=.6, label="AlphaMissense (post-2023)")
    for xi, m in zip(x, models):
        ax.annotate(m.split("_")[1], (xi, sc[m]["auroc_all"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set(xscale="log", xlabel="ESM-2 parameters", ylabel="AUROC", title="Zero-shot accuracy scales with model size")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "scaling.png"); plt.close(fig)

    # 3. per-gene AUROC scatter(s)
    pairs = [("ESM-2 650M", "zero-shot ESM-2")] + ([("VariantLens+AM", "VariantLens+AM")] if "VariantLens+AM" in pg else [])
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    for ax, (k, lab) in zip(axes[0], pairs):
        better = int((pg[k] > pg["AlphaMissense"]).sum())
        ax.scatter(pg["AlphaMissense"], pg[k], s=np.clip(pg.n / 4, 8, 120), alpha=.55, c=COL[k], lw=0)
        ax.plot([0.3, 1], [0.3, 1], c="grey", ls=":")
        for _, r in pg.assign(d=(pg[k] - pg["AlphaMissense"]).abs()).nlargest(7, "d").iterrows():
            ax.annotate(r.gene, (r["AlphaMissense"], r[k]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="AlphaMissense per-gene AUROC", ylabel=f"{lab} per-gene AUROC",
               title=f"{lab} better in {better} / {len(pg)} genes")
        ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.01)
    fig.tight_layout(); fig.savefig(FIG / "per_gene.png"); plt.close(fig)

    # 4. score distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, c, n in [(axes[0], best, "ESM-2 650M log-likelihood ratio"), (axes[1], "am_pathogenicity", "AlphaMissense pathogenicity")]:
        for lab, name, col in [(0, "Benign / LB", "#81b29a"), (1, "Pathogenic / LP", "#e07a5f")]:
            ax.hist(df.loc[df.label == lab, c], bins=80, alpha=.6, color=col, label=name, density=True)
        ax.set(xlabel=n, ylabel="density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "distributions.png"); plt.close(fig)

    # 5. ESM vs AM scatter
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    s = df.sample(min(len(df), 15000), random_state=0)
    for lab, col in [(0, "#81b29a"), (1, "#e07a5f")]:
        x = s[s.label == lab]
        ax.scatter(x.am_pathogenicity, x[best], s=3, alpha=.35, c=col, lw=0, label=["Benign", "Pathogenic"][lab])
    ax.set(xlabel="AlphaMissense", ylabel="ESM-2 650M LLR", title="Two views of the same variants")
    ax.legend(frameon=False, markerscale=4); fig.tight_layout(); fig.savefig(FIG / "esm_vs_am.png"); plt.close(fig)


if __name__ == "__main__":
    main()
