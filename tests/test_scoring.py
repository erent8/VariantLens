import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from variantlens.esm_scoring import MAX_RESIDUES, ESMScorer, parse_variant  # noqa: E402


def test_parse_variant():
    assert parse_variant("R175H") == ("R", 174, "H")


@pytest.mark.parametrize("i,L", [(0, 500), (499, 500), (0, 5000), (2500, 5000), (4999, 5000)])
def test_window_contains_position(i, L):
    s = ESMScorer.window_start(i, L)
    assert 0 <= s <= max(0, L - MAX_RESIDUES)
    assert s <= i < s + MAX_RESIDUES


@pytest.fixture(scope="module")
def tiny():
    return ESMScorer("esm2_8M", device="cpu", half=False)


def test_wt_marginal_shape_and_zero_diagonal(tiny):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    L = tiny.wt_marginal(seq)
    assert L.shape == (len(seq), 20)
    idx = [ "ACDEFGHIKLMNPQRSTVWY".index(a) for a in seq]
    assert np.allclose(L[np.arange(len(seq)), idx], 0)


def test_long_sequence_windows_match_short(tiny):
    """For a long protein, a position scored inside its centred window should
    equal scoring that window on its own."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 1500))
    mm = tiny.masked_marginal(seq, [1400])[1400]
    s = ESMScorer.window_start(1400, len(seq))
    ref = tiny.masked_marginal(seq[s:s + MAX_RESIDUES], [1400 - s])[1400 - s]
    assert np.allclose(mm, ref, atol=1e-4)
