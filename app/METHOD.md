### What this is

A protein language model (ESM-2) has only ever seen raw protein sequences — no structures, no clinical labels, no
population genetics. Yet by learning which amino acids evolution tolerates at each position, it implicitly learns
which mutations break proteins. VariantLens measures how far that *zero-shot* signal goes on real human disease
variants, head-to-head against DeepMind's AlphaMissense.

### Scoring

For a missense change *wt → mt* at position *i*:

* **Masked marginal** (benchmark): replace residue *i* with `<mask>` and compute
  `LLR = log p(mt | x₋ᵢ) − log p(wt | x₋ᵢ)`. One forward pass per position.
* **Wild-type marginal** (landscapes): one pass over the unmasked sequence gives all *L × 19* substitutions at once.

Negative LLR → the model finds the mutant less plausible than the wild type → likely damaging.
Proteins longer than 1,022 residues are scored with sliding windows centred on each position.

### Data

* **Labels:** ClinVar SNVs classified Pathogenic/Likely pathogenic vs Benign/Likely benign with
  ≥1 review star; conflicting interpretations removed.
* **Mapping:** joined to AlphaMissense on GRCh38 coordinates (chrom, pos, ref, alt), giving the canonical
  UniProt isoform and protein change; variants whose reference residue mismatches the current UniProt sequence are dropped.
* **Temporal hold-out:** variants that had no P/LP/B/LB classification in the **August 2023** ClinVar release
  (AlphaMissense publication) form the *post-2023* split, which neither model could have seen in any form.

### Evaluation

Global AUROC/AUPRC (95 % bootstrap CIs), per-gene AUROC on genes with ≥5 pathogenic and ≥5 benign variants
(the clinically relevant "rank variants within a gene" task), an ESM-2 scaling series (8M → 650M parameters),
and a two-feature ensemble trained with gene-grouped cross-validation so no gene is in both train and test.

### Caveats

* AlphaMissense's score thresholds were calibrated with ClinVar, so on the *all* split it has a mild
  home advantage; the post-2023 split removes it.
* ClinVar is biased toward well-studied genes and toward variants that are easy to classify.
* Scores are research tools, **not** clinical diagnoses.
