"""Inference with the trained VariantLens meta-predictors."""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from .esm_scoring import AA
from .features import landscape_features

RES = Path(__file__).resolve().parents[2] / "results"


@lru_cache(maxsize=None)
def load(name: str):
    f = RES / f"model_{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def all_variants(seq: str) -> list[str]:
    return [f"{w}{i + 1}{m}" for i, w in enumerate(seq) if w in AA for m in AA if m != w]


def predict(L: np.ndarray, seq: str, variants: list[str], am: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """P(pathogenic) for each variant from VariantLens-Seq and, when AlphaMissense
    scores are supplied, VariantLens+AM (NaN where AM has no score)."""
    X = landscape_features(L, seq, variants)
    out = {}
    m = load("vl_seq")
    if m:
        out["vl_seq"] = m["model"].predict_proba(X[m["features"]])[:, 1]
    m = load("vl_am")
    if m and am is not None:
        X["am_pathogenicity"] = [am.get(v, np.nan) for v in variants]
        p = m["model"].predict_proba(X[m["features"]])[:, 1]
        p[X.am_pathogenicity.isna().values] = np.nan
        out["vl_am"] = p
    return out
