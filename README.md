# AbDock-AI — Antibody–Antigen Binding Affinity Prediction

A machine-learning pipeline that predicts **antibody–antigen binding affinity (pKD)** from
sequence and 3D structure. Binding affinity is the central quantity optimized during
**antibody affinity maturation** — the lead-optimization step of antibody drug discovery —
so a model that ranks binders reliably can reduce costly wet-lab iterations.

> **Headline result:** Spearman **0.39–0.41** on a *leakage-free antigen-cluster split*
> (held-out antigens the model never saw during training), up from ≈0 for the initial
> pipeline. Paired bootstrap testing shows the embedding variants are **statistically
> indistinguishable** on a 150-sample test set — a finding this project reports rather
> than papers over. The emphasis here is a **rigorous, honestly-evaluated** pipeline
> rather than an inflated benchmark number.

**Stack:** Python · PyTorch · ESM-2 (650M) · XGBoost · scikit-learn · BioPython · pandas/NumPy

---

## Why this is hard (and why the evaluation matters)

Predicting *absolute* binding affinity across *diverse* antigens from sequence is one of the
hardest tasks in the field. The common trap is to evaluate on a **random split**, where
near-identical antibodies leak between train and test and inflate the score. This project
found and corrected exactly that failure mode:

| Evaluation setting | Reported R² | Verdict |
| --- | --- | --- |
| Random split (naïve) | ~0.50 | ❌ Data leakage — antibody families shared across train/test |
| Antigen-cluster split (honest) | 0.16 (Spearman **0.41**) | ✅ Reflects true generalization to unseen antigens |

Knowing the gap between these two numbers is the difference between a demo and a
deployable model.

---

## Method

```mermaid
flowchart LR
    A[SAbDab complexes<br/>~1,100 curated] --> B[Data curation<br/>pKD labels · dedup<br/>antigen-cluster split]
    B --> C1[ESM-2 650M<br/>whole-chain mean pooling<br/>VH · VL · antigen]
    B --> C2[PDB/CIF structures<br/>BioPython]
    C2 --> D1[Interface detection<br/>paratope / epitope<br/>CA contacts]
    D1 --> C3[ESM-2 interface-residue<br/>pooling]
    C2 --> D2[Geometric features<br/>contacts · VH–VL geom]
    C1 --> E[Feature fusion + PCA]
    C3 --> E
    D2 --> E
    E --> F[XGBoost / Ridge<br/>GroupKFold CV by antigen]
    F --> G[Predicted pKD<br/>+ honest ranking metrics]
```

**Two complementary sequence representations**

- **Whole-chain mean pooling** — mean of ESM-2 per-residue embeddings over the full VH, VL,
  and antigen chains.
- **Interface-residue pooling** — ESM-2 embeddings pooled over *only* the paratope
  (antibody residues at the interface) and epitope (antigen residues at the interface),
  identified from the 3D structure. Sequence and structure indices are kept naturally
  aligned, so no external numbering map is required.

**Structural features** — interface residue/contact counts (CA < 8 Å, heavy < 5 Å),
chain sizes, and VH–VL geometry, extracted from PDB/CIF with BioPython.

---

## Results

Honest evaluation on the **antigen-cluster test split**, after de-duplication
(758 unique complexes → 608 train / 150 test), XGBoost + PCA(50). 95% CIs from
8,000 bootstrap resamples; Δ is the paired difference vs. whole-chain pooling.

| Feature set | Spearman | 95% CI | Δ vs. baseline |
| --- | :---: | :---: | :---: |
| Structural features only | 0.167 | [0.02, 0.31] | −0.224 **(sig.)** |
| Interface pooling, H/L separate | 0.195 | [0.05, 0.34] | −0.198 **(sig.)** |
| Interface pooling (paratope + epitope) | 0.251 | [0.10, 0.39] | −0.141 (n.s.) |
| Interface pooling, junction-free | 0.293 | [0.14, 0.44] | −0.098 (n.s.) |
| Mean pooling + structural features | 0.375 | [0.23, 0.51] | −0.018 (n.s.) |
| **Whole-chain mean pooling (VH+VL+antigen)** | **0.393** | [0.25, 0.53] | *baseline* |
| Mean pooling + interface pooling (fused) | 0.413 | [0.27, 0.54] | +0.020 (n.s.) |

**Findings**

- The largest gain came from **fixing evaluation and data hygiene** (a prediction-alignment
  bug, sequence-based de-duplication, and stronger regularization), which moved the honest
  held-out correlation from ≈0 to ~0.39.
- **Only 2 of 8 pairwise comparisons are statistically significant**, and both are
  *negative*: structural features alone, and splitting H/L into separate vectors. With 150
  test samples the resolution limit on Spearman is ≈±0.17, so the apparent +0.020 from
  fusion is **noise, not signal** — the grouped-CV estimate (n=608, 4× the samples) actually
  ranks plain whole-chain pooling highest.
- A tested-and-refuted hypothesis: concatenating heavy and light chains into a single ESM
  input creates an artificial junction. Removing it (holding dimensionality constant) gave
  Δ = +0.040, CI [−0.088, +0.169] — **not significant**.
- Hand-crafted structural features add nothing on top of language-model embeddings, which is
  consistent with protein language models implicitly encoding structural contacts.

---

## Repository layout

```
AIDD/
├── data/          # SAbDab download + CIF structures (splits_final/)
├── parser/        # VH/VL chain & sequence extraction from PDB
├── scripts/
│   ├── 01_build_dataset.py                # SKEMPI v2 branch (mutation ΔΔG pairs)
│   ├── 01b_build_sabdab2_dataset.py       # curate labeled complexes (pKD, splits)
│   ├── 02_extract_structural_features.py  # BioPython interface/geometry features
│   ├── 04_extract_esm2_embeddings.py      # ESM-2 whole-chain mean embeddings
│   ├── 05_train_esm_xgb.py                # trainer: fusion, PCA, GroupKFold CV
│   ├── 06_extract_interface_embeddings.py # ESM-2 paratope/epitope pooling
│   ├── 07_run_ablations.py                # one-command ablation suite → table
│   └── 08_significance_test.py            # bootstrap CIs + paired significance
├── processed/     # feature tables & embeddings (.npz)
└── docs/plan.md   # step-by-step build log
```

---

## Reproduce

```bash
conda activate aidd

# Full ablation suite → reports/ablation_table.md
python AIDD/scripts/07_run_ablations.py

# Bootstrap CIs + paired significance → reports/significance_table.md
python AIDD/scripts/08_significance_test.py --ref mean_pooled

# Single best-observed configuration
python AIDD/scripts/05_train_esm_xgb.py \
  --emb-npz AIDD/processed/esm2_650m_embeddings.npz \
            AIDD/processed/esm2_interface_embeddings.npz \
  --feature-keys combined_embeddings paratope_embeddings epitope_embeddings \
  --model xgb --pca 50
```

Upstream artifacts (labels, structural features, embeddings) are produced by scripts
`01b`, `02`, `04`, and `06` respectively; the trainer (`05`) is the fast, iterate-often step.

---

## Rigor highlights (what this project demonstrates)

- **Leakage-aware evaluation:** antigen-clustered train/test split + `GroupKFold` CV grouped
  by antigen, so validation reflects generalization to unseen antigens.
- **Data hygiene:** de-duplication of near-identical complex copies before splitting.
- **Ranking-first metrics:** Spearman/Pearson reported alongside R², because absolute
  affinity calibration across antigens is not realistic — ranking is the useful signal.
- **Uncertainty quantification:** paired bootstrap testing on every ablation, which revealed
  that most of the apparent differences — including one this project initially claimed as a
  gain — are within sampling noise.
- **Debugging:** identified and fixed a prediction-alignment bug that had masked the model's
  true behavior.

## Limitations & next steps

- The ceiling for cross-antigen *absolute* pKD on this data is ~Spearman 0.4; labels mix
  assay types and confidence levels, which caps achievable accuracy.
- **The test set is too small to compare representations.** At n=150 the resolution limit is
  ≈±0.17 Spearman. Distinguishing embedding strategies would need either a much larger
  labeled set or a paired within-complex task.
- A more tractable and directly useful reframing is **ΔΔG mutation ranking** — predicting
  which point mutations improve binding within a single antibody lineage (the actual
  affinity-maturation task), which pairs naturally with the planned genetic-algorithm
  optimizer.
- Swapping generic ESM-2 for an **antibody-specific language model** (AntiBERTy / IgBERT) is
  a promising next experiment for the VH/VL representations.

---

*Data: [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/). Protein language model:
[ESM-2](https://github.com/facebookresearch/esm).*
