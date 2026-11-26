"""Fetch UniProt sequences for benchmark proteins and drop variants whose
reference residue does not match the current canonical sequence."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
URL = "https://rest.uniprot.org/uniprotkb/accessions"


def fetch(accs: list[str]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for i in range(0, len(accs), 200):
        batch = accs[i : i + 200]
        for attempt in range(5):
            r = requests.get(URL, params={"accessions": ",".join(batch), "format": "fasta"}, timeout=120)
            if r.ok:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        acc, buf = None, []
        for line in r.text.splitlines() + [">"]:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc = line.split("|")[1] if "|" in line else None
                buf = []
            else:
                buf.append(line.strip())
        print(f"  fetched {len(seqs):,}/{len(accs):,}", flush=True)
    return seqs


def main():
    df = pd.read_parquet(PROC / "benchmark.parquet")
    accs = sorted(df.uniprot.unique())
    seqs = fetch(accs)
    pd.DataFrame({"uniprot": list(seqs), "sequence": list(seqs.values())}).to_parquet(
        PROC / "sequences.parquet", index=False)

    def ok(row):
        s = seqs.get(row.uniprot)
        if s is None:
            return False
        p = int(row.variant[1:-1]) - 1
        return p < len(s) and s[p] == row.variant[0]

    df["ref_match"] = df.apply(ok, axis=1)
    print(f"reference residue matches: {df.ref_match.mean():.3%}")
    df = df[df.ref_match].drop(columns="ref_match")
    df.to_parquet(PROC / "benchmark.parquet", index=False)
    print(f"kept {len(df):,} variants in {df.uniprot.nunique():,} proteins")


if __name__ == "__main__":
    main()
