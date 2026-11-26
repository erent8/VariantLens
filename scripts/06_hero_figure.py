"""README hero figure: the full predicted mutational landscape of a protein with
ClinVar variants overlaid.   python scripts/06_hero_figure.py [GENE]"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from variantlens.esm_scoring import AA_IDX  # noqa: E402

ORDER = "DEKRHNQSTYCGAVLIMPFW"
gene = sys.argv[1] if len(sys.argv) > 1 else "TP53"
prot = pd.read_parquet(ROOT / "app/data/proteins.parquet")
row = prot[prot.gene == gene].iloc[0]
var = pd.read_parquet(ROOT / "app/data/variants.parquet")
clin = var[var.uniprot == row.uniprot]
L = np.load(ROOT / f"data/processed/landscapes/esm2_650M/{row.uniprot}.npy").astype(np.float32)
Z = L[:, [AA_IDX[a] for a in ORDER]].T
n = len(row.sequence)

cmap = LinearSegmentedColormap.from_list("vl", ["#8c1c13", "#e07a5f", "#f4d8c8", "#f7f7f7", "#4393c3"])
fig = plt.figure(figsize=(16, 5.6), dpi=170)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.2], width_ratios=[60, 1], hspace=0.06, wspace=0.01)
ax0, ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

sens = -np.nanmean(np.where(L == 0, np.nan, L), axis=1)
ax0.fill_between(np.arange(1, n + 1), sens, color="#e07a5f", alpha=.35, lw=0)
ax0.plot(np.arange(1, n + 1), sens, color="#c2553a", lw=.8)
ax0.set_xlim(.5, n + .5); ax0.set_xticks([]); ax0.set_ylabel("intolerance", fontsize=9)
for s in ("top", "right", "bottom"):
    ax0.spines[s].set_visible(False)
ax0.set_title(f"{gene}: all {n * 19:,} possible missense changes scored by ESM-2 (zero-shot), "
              f"with {len(clin)} ClinVar variants overlaid", loc="left", fontsize=12, fontweight="bold")

im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=-14, vmax=2.5, interpolation="nearest",
               extent=[.5, n + .5, 19.5, -.5])
ax.set_yticks(range(20)); ax.set_yticklabels(list(ORDER), fontsize=7, family="monospace")
ax.set_xlabel("residue"); ax.set_ylabel("substituted amino acid")
for lab, m, c, nm in [(1, "x", "#111111", "ClinVar pathogenic / likely pathogenic"),
                      (0, "o", "#1b7f4b", "ClinVar benign / likely benign")]:
    d = clin[clin.label == lab]
    ax.scatter([int(v[1:-1]) for v in d.variant], [ORDER.index(v[-1]) for v in d.variant], marker=m,
               s=14, c=c if m == "x" else "none", edgecolors=c, linewidths=.9, label=nm)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, fontsize=9)
cb = fig.colorbar(im, cax=cax); cb.set_label("ESM-2 log-likelihood ratio", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
out = ROOT / "results/figures" / f"landscape_{gene}.png"
fig.savefig(out, bbox_inches="tight"); print(out)
