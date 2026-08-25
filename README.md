# Explainable Learning Analytics for At-Risk Cybersecurity Training Prediction in Ghanaian Banks Using XGBoost-SHAP

Reference implementation accompanying the Methods and Results sections
submitted for this study (Student: Alex Tawiah Ampofo, 22527933;
Supervisor: Dr. Eric Opoku Osei).

**Model status:** ENGINEERED — single REPLACE operation (standard
logistic loss substituted with a focal-loss objective; see
`src/focal_loss.py` and Methods M11–M12).

**Dataset strategy:** CROSS-DATASET (Methods M10) — the primary field
corpus and the public IBM HR validation corpus are kept strictly
separate at every stage. No row, feature value, fitted transform, or
trained weight is ever shared between them.

## Repository structure

```
.
├── src/
│   ├── focal_loss.py       # Algorithm 1 / Methods M12 objective (shared)
│   ├── preprocessing.py    # Shared Procedure A / Procedure B utilities
│   ├── field_pipeline.py   # Primary corpus (field) pipeline
│   └── ibm_pipeline.py     # Validation corpus (IBM HR) pipeline
├── notebooks/
│   └── xgboost_shap_pipeline.ipynb   # Single-cell Colab-friendly notebook
├── data/
│   └── README.md           # Data access notes (see below)
├── results/                # Output JSON summaries land here when scripts are run
├── requirements.txt
└── LICENSE
```

## The two evaluation procedures (Methods M9)

Every corpus is evaluated with two separate, **non-nested** procedures:

- **Procedure A** — repeated stratified 5-fold cross-validation, 3
  repeats (15 folds total), applied directly to the full corpus. This
  produces the headline mean ± SD results (Results Tables 4, 5, 7).
- **Procedure B** — a single stratified 80:20 train:test split, drawn
  once and independently of Procedure A. This produces the single-model
  illustrative diagnostics only (Results Figure 4, Figure 6, Figure 7,
  Table 6). Procedure B's test partition is never used to compute the
  Procedure A headline metrics.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipelines

**Validation corpus (public, no setup required):**

```bash
python src/ibm_pipeline.py
```

Downloads the IBM HR Analytics Employee Attrition dataset automatically
from its public mirror and writes `results/ibm_results.json`.

**Primary corpus (requires your own field data copy):**

```bash
python src/field_pipeline.py --data path/to/your_field_survey.xlsx
```

The raw field questionnaire export is **not included** in this
repository (see `data/README.md` and Methods M5/M21 for the
data-governance reasons). Supply your own de-identified copy locally
with the same column structure to reproduce the primary-corpus results.

## Reproducibility note

This repository is a cleaned-up, modularised reimplementation of the
exploratory pipeline originally used to produce the numbers reported in
the Methods and Results documents. The IBM (validation-corpus) numbers
reproduce the published tables almost exactly. The field
(primary-corpus) numbers reproduce closely (typically within
0.005–0.01 AUC-ROC) but are not guaranteed bit-identical, due to minor
random-state consumption differences introduced when the original
scripts were refactored into the shared `preprocessing.py` module. The
published Methods/Results tables should be treated as authoritative;
this code is provided for methodological transparency and to allow
independent verification of the general approach, not as a guarantee of
decimal-exact replication.

A single fixed random seed (`SEED = 42`) governs data partitioning,
SMOTE resampling, and each model's own internal stochastic elements
consistently across all folds (Methods M14). As noted in M14, this
provides evidence of variability across data partitions but does not by
itself constitute a multi-seed protocol varying the model's own training
stochasticity independently of the partition.

## Hyperparameters

The specific hyperparameter values used by the proposed model (selected
by an Optuna tree-structured Parzen estimator search, Methods M13) are
hard-coded in `FIELD_CFG` (`src/field_pipeline.py`) and `IBM_CFG`
(`src/ibm_pipeline.py`) for exact reproducibility of the reported
configuration, rather than re-run at import time.

## Citation

If you use this code, please cite the accompanying thesis / manuscript
(citation details to be added upon publication).

## Licence

MIT — see `LICENSE`. The public IBM HR Analytics dataset retains its own
original licence terms (Methods Table 2 / M4).
