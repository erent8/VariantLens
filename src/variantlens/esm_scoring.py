"""Zero-shot variant effect scoring with ESM-2 protein language models.

Two standard scoring strategies (Meier et al., 2021, "Language models enable
zero-shot prediction of the effects of mutations on protein function"):

* wild-type marginal (WT):  one forward pass on the unmasked sequence,
  score(i, mt) = log p(mt | x) - log p(wt | x)  at position i.
  Cheap -> used to build full mutational landscapes (L x 20).

* masked marginal (MM): position i is replaced by <mask>,
  score(i, mt) = log p(mt | x_-i) - log p(wt | x_-i).
  One forward pass per position; slightly more accurate -> used for benchmark.

Negative scores = the substitution is less "plausible" to the model than the
wild type = more likely deleterious.

ESM-2 has a 1024-token context (1022 residues + BOS/EOS). Longer proteins are
handled with sliding windows, choosing for every position the window in which
it is most central.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
MAX_RESIDUES = 1022

MODELS = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
}


@dataclass
class ESMScorer:
    name: str = "esm2_650M"
    device: str | None = None
    half: bool = True
    token_budget: int = 12_000  # residues per batch for masked-marginal passes

    def __post_init__(self):
        from transformers import AutoTokenizer, EsmForMaskedLM

        hf_id = MODELS.get(self.name, self.name)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        model = EsmForMaskedLM.from_pretrained(hf_id)
        if self.half and self.device == "cuda":
            model = model.half()
        self.model = model.to(self.device).eval()
        self.aa_tok = torch.tensor(
            [self.tok.convert_tokens_to_ids(a) for a in AA], device=self.device
        )
        self.mask_id = self.tok.mask_token_id

    # ------------------------------------------------------------------ utils
    def _encode(self, seq: str) -> torch.Tensor:
        return self.tok(seq, return_tensors="pt")["input_ids"][0]

    @staticmethod
    def window_start(i: int, L: int, W: int = MAX_RESIDUES) -> int:
        """Start of the length-W window that centres residue i (0-based)."""
        if L <= W:
            return 0
        return int(min(max(i - W // 2, 0), L - W))

    @torch.no_grad()
    def _log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> log-probs over the 20 amino acids, (B, T, 20), float32."""
        out = self.model(input_ids=ids.to(self.device)).logits.float()
        return torch.log_softmax(out, dim=-1)[..., self.aa_tok]

    # --------------------------------------------------- wild-type marginal
    def wt_marginal(self, seq: str) -> np.ndarray:
        """Full (L, 20) landscape of log-likelihood ratios, wild-type marginal."""
        L = len(seq)
        out = np.zeros((L, 20), dtype=np.float32)
        wt_idx = np.array([AA_IDX.get(a, -1) for a in seq])
        # group positions by the window that centres them
        # windows at stride W/4: every residue is at most W/8 from its window centre
        # (or at a true sequence end), at ~4x the cost of a single pass
        starts = [0] if L <= MAX_RESIDUES else sorted(
            set(range(0, L - MAX_RESIDUES + 1, MAX_RESIDUES // 4)) | {L - MAX_RESIDUES})
        centre = np.array(starts) + MAX_RESIDUES / 2
        assign = np.argmin(np.abs(np.arange(L)[:, None] - centre[None, :]), axis=1) if L > MAX_RESIDUES else np.zeros(L, int)
        for w, s in enumerate(starts):
            pos = np.where(assign == w)[0]
            if len(pos) == 0:
                continue
            sub = seq[s : s + MAX_RESIDUES]
            lp = self._log_probs(self._encode(sub)[None])[0, 1:-1].cpu().numpy()
            out[pos] = lp[pos - s]
        valid = wt_idx >= 0
        out[valid] -= out[valid, wt_idx[valid]][:, None]
        out[~valid] = np.nan
        return out

    # ------------------------------------------------------ masked marginal
    def masked_marginal(self, seq: str, positions) -> dict[int, np.ndarray]:
        """Masked-marginal LLR vectors (20,) for the given 0-based positions."""
        L = len(seq)
        positions = sorted(set(int(p) for p in positions))
        by_window: dict[int, list[int]] = {}
        for p in positions:
            by_window.setdefault(self.window_start(p, L), []).append(p)
        res: dict[int, np.ndarray] = {}
        for s, pos in by_window.items():
            sub = seq[s : s + MAX_RESIDUES]
            base = self._encode(sub)
            bs = max(1, self.token_budget // len(sub))
            for k in range(0, len(pos), bs):
                chunk = pos[k : k + bs]
                ids = base.repeat(len(chunk), 1)
                rows = torch.arange(len(chunk))
                cols = torch.tensor([p - s + 1 for p in chunk])  # +1 for BOS
                ids[rows, cols] = self.mask_id
                lp = self._log_probs(ids)[rows, cols.to(self.device)].cpu().numpy()
                for p, v in zip(chunk, lp):
                    wt = AA_IDX.get(seq[p])
                    res[p] = v - v[wt] if wt is not None else np.full(20, np.nan, np.float32)
        return res


def parse_variant(v: str) -> tuple[str, int, str]:
    """'R175H' -> ('R', 174, 'H')  (0-based position)."""
    return v[0], int(v[1:-1]) - 1, v[-1]
