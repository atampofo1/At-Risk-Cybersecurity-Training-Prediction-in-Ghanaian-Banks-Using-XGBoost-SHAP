# Explainable Learning Analytics for At-Risk Cybersecurity Training Prediction in Ghanaian Banks Using XGBoost-SHAP

Reference implementation accompanying the Methods and Results sections
(Student: Alex Tawiah Ampofo, 22527933; Supervisor: Dr. Eric Opoku Osei).
**This revision corrects a data-quality bug and a nested-cross-validation
leakage pattern found during a structured diagnostic audit; see
"Revision history and honest findings" below before using any number
from an earlier version of this repository.**

**Model status:** ENGINEERED — single REPLACE operation (standard
logistic loss substituted with a focal-loss objective; see
`src/focal_loss.py` and Methods M11–M12). **This substitution did not
produce a statistically significant improvement over a properly tuned
baseline on either corpus in this revision's diagnostic re-run** — see
below.

**Dataset strategy:** CROSS-DATASET (Methods M10) — the primary field
corpus and the public IBM HR validation corpus are kept strictly
separate at every stage.

## Revision history and honest findings

A structured diagnostic audit (following the CAN-DO Lab negative-results
protocol) found and fixed two issues present in an earlier version of
this code, then ran a complete corrected re-analysis. Both the bugs and
the corrected findings are reported here in full, not just the fix.

### Bug 1 — data-quality: informative missingness miscoded as zero

The primary corpus's training-session-count item had **33.2% non-response
(177 of 533 records)**, and no respondent selected a genuine "zero
sessions" category at all — the lowest option anyone actually chose was
"1 session." The original pipeline (`fillna(0)`) silently coded every
non-response as zero sessions, which is factually wrong, and the
non-response pattern turned out to be highly informative: non-responders
were at-risk **96.0%** of the time versus **74.2%** for responders.

**Fix:** the item is now split into a genuine-response ordinal feature
(non-response left as `NaN`, never coded to zero) and an explicit binary
non-response indicator, both defined in `field_pipeline.py`.

### Bug 2 — nested cross-validation leakage (GATE-1)

Feature selection (SHAP + RFE) and hyperparameter tuning were previously
fitted once, outside the cross-validation loop, then reused across all
folds — an inner tuning partition was never defined. **Fix:** both are
now fitted on an inner-training partition within each outer fold only,
with the resulting configuration scored exactly once on that fold's
outer-test portion (see `src/full_diagnostic_battery.py` and Methods M9).
Random Forest, Logistic Regression, and SVM baselines were also given a
matched tuning budget, closing a separate baseline-symmetry gap (GATE-2).

### Corrected headline finding

Under the corrected, leakage-free, matched-budget nested design, **the
focal-loss substitution's AUC-ROC advantage over the baseline was not
statistically significant on either corpus** (field p = 0.070, validation
p = 0.300), and it did not outperform a properly tuned Random Forest on
either corpus. The full ablation attributes the small observed gain
almost entirely to independent hyperparameter tuning, not to the loss
substitution itself. A related but distinct intervention — a
class-weighted logistic loss, with the weight itself tuned — **did**
achieve a significant, replicated improvement on both corpora (field
p = 0.048, validation p = 0.009). This is reported as the honest
alternative finding, not substituted retroactively for the declared
engineering operation. Full detail is in Results Section R10.

## Repository structure

```
.
├── src/
│   ├── focal_loss.py              # Algorithm 1 / Methods M12 objective (shared)
│   ├── preprocessing.py           # Shared utilities + per-fold imputation
│   ├── field_pipeline.py          # Primary corpus pipeline (corrected encoding)
│   ├── ibm_pipeline.py            # Validation corpus pipeline
│   └── full_diagnostic_battery.py # Complete corrected diagnostic battery
│                                   # (nested CV, matched-budget comparators,
│                                   # threshold sweep, SMOTE x focal interaction,
│                                   # seed-variance decomposition, re-engineering)
├── notebooks/
│   └── xgboost_shap_pipeline.ipynb   # Legacy single-cell notebook (see note inside)
├── data/
│   └── README.md
├── results/
├── requirements.txt
└── LICENSE
```

## The two evaluation procedures (Methods M9)

Every corpus is evaluated with two separate, **non-nested** procedures:

- **Procedure A** — repeated stratified 5-fold cross-validation, 3
  repeats (15 outer folds). Within each outer fold, the outer-training
  portion is further split into an inner-train/inner-validation
  partition; feature selection and hyperparameter tuning are both fitted
  on the inner-training partition only, never on the outer-test fold.
  Produces the headline mean ± SD results (Results Tables 4, 5, 7).
- **Procedure B** — a single stratified 80:20 train:test split, drawn
  once and independently of Procedure A. Produces the single-model
  illustrative diagnostics only (Results Figure 4, Figure 6, Figure 7,
  Table 6).

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

**Primary corpus (requires your own field data copy):**

```bash
python src/field_pipeline.py --data path/to/your_field_survey.xlsx
```

The raw field questionnaire export is **not included** in this
repository (see `data/README.md` and Methods M5/M21).

**Full diagnostic battery** (nested CV + matched-budget RF/LR/SVM +
threshold sweep + SMOTE×focal interaction + seed-variance decomposition
+ two re-engineering attempts, on either corpus):

```python
import pandas as pd, json
from full_diagnostic_battery import run_battery
# see run_battery()'s docstring for the expected DataFrame shape
```

## Reproducibility note

A fixed random seed (`SEED = 42`) governs data partitioning, SMOTE
resampling, and each model's own internal stochastic elements
consistently across all folds (Methods M14). A supplementary
seed-variance decomposition (`full_diagnostic_battery.py`, Part 4) found
that variance across data partitions is 500–800× larger than variance
across independent model-training seeds on this model and dataset
combination, so this single-seed design understates total variance only
marginally.

Inner-fold hyperparameter search uses a reduced trial budget (8–11 trials
per outer fold, stated explicitly in Methods M13) rather than a nominal
50-trial budget, for tractability across fifteen outer folds × five
model families × two corpora. This is a compute-budget limitation, not a
methodological choice, and is disclosed as such.

## Citation

If you use this code, please cite the accompanying thesis / manuscript
(citation details to be added upon publication).

## Licence

MIT — see `LICENSE`. The public IBM HR Analytics dataset retains its own
original licence terms.
