"""Variant features derived from an ESM-2 mutational landscape (L x 20 LLR matrix).

The zero-shot LLR only looks at a single cell of the landscape. The landscape
around it carries more information: how intolerant the whole position is, how
intolerant its sequence neighbourhood is (a proxy for buried / functional
regions), and where the substitution ranks among the 19 alternatives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .esm_scoring import AA, AA_IDX

# amino-acid physico-chemistry (Kyte-Doolittle hydropathy, residue volume in A^3)
HYDRO = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
VOLUME = dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))

FEATURES = ["llr", "pos_mean", "pos_min", "pos_std", "mt_rank", "nbr5_mean", "nbr15_mean",
            "rel_pos", "log_len", "d_hydro", "d_volume", "wt_aa", "mt_aa"]
CATEGORICAL = ["wt_aa", "mt_aa"]


def landscape_features(L: np.ndarray, seq: str, variants: list[str]) -> pd.DataFrame:
    n = len(L)
    sub = L.astype(np.float32).copy()
    wt_idx = np.array([AA_IDX.get(a, -1) for a in seq])
    ok = wt_idx >= 0
    sub[np.where(ok)[0], wt_idx[ok]] = np.nan  # exclude wild-type column (LLR == 0 by definition)
    pos_mean = np.nanmean(sub, axis=1)
    pos_min = np.nanmin(sub, axis=1)
    pos_std = np.nanstd(sub, axis=1)
    kernel5, kernel15 = np.ones(11) / 11, np.ones(31) / 31
    pm = np.nan_to_num(pos_mean)
    nbr5 = np.convolve(pm, kernel5, mode="same")
    nbr15 = np.convolve(pm, kernel15, mode="same")
    wt = np.array([AA_IDX[v[0]] for v in variants])
    p = np.array([int(v[1:-1]) - 1 for v in variants])
    m = np.array([AA_IDX[v[-1]] for v in variants])
    llr = L[p, m].astype(np.float32)
    hyd, vol = np.array([HYDRO[a] for a in AA]), np.array([VOLUME[a] for a in AA])
    return pd.DataFrame({
        "llr": llr, "pos_mean": pos_mean[p], "pos_min": pos_min[p], "pos_std": pos_std[p],
        "mt_rank": np.nansum(sub[p] < llr[:, None], axis=1) / 19, "nbr5_mean": nbr5[p], "nbr15_mean": nbr15[p],
        "rel_pos": p / n, "log_len": np.full(len(p), np.log(n)), "d_hydro": hyd[m] - hyd[wt],
        "d_volume": vol[m] - vol[wt], "wt_aa": wt, "mt_aa": m,
    })
