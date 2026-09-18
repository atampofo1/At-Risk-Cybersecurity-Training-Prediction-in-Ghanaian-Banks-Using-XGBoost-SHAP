# Explainable Learning Analytics for At-Risk Cybersecurity Training Prediction in Ghanaian Banks Using XGBoost-SHAP

Reference implementation for the Methods and Results sections (Student:
Alex Tawiah Ampofo, 22527933; Supervisor: Dr. Eric Opoku Osei).

## Headline finding (Results R10)

Under a corrected, leakage-free nested cross-validation with a matched
inner-fold tuning budget across all comparators, the proposed focal-loss
substitution does **not** produce a statistically significant improvement
over a properly tuned baseline on either corpus (field: dAUC +0.0040,
p = 0.140; validation: dAUC +0.0032, p = 0.524). Two re-engineering
attempts (an interaction feature; a class-weighted loss) also fail once
their own selection-on-test defects are corrected. This is reported as a
diagnosed negative result, not concealed.

## Corrections applied in this revision (verification pass K1-K5)

- **K1** Parts 2-5 of the diagnostic battery use the full unselected
  candidate-feature set (no selection outside a fold; nothing to leak).
- **K2/K2b/K2c** Every tuned arm (baseline, proposed, RF, LR, SVM)
  receives an equal inner-fold trial budget; LR and SVM are now genuinely
  tuned rather than left at defaults. Search-space dimensionality still
  differs by model architecture and is stated as a limitation in Methods
  M13 (Q10/C4).
- **K3** The class-weighted-loss weight is calibrated on a disjoint fold
  subset from the one it is scored on.
- **K4** All Wilcoxon significance tests are two-sided.
- **K5** Per-fold gamma/alpha are retained and emitted
  (`part1_focal_params`).
- **Data-quality (Methods M6)** The training-session item's 33.2%
  non-response is no longer miscoded as zero; it is split into a
  genuine-response ordinal (imputed per-fold) plus an explicit
  missingness indicator.

## Structure

```
src/focal_loss.py               Algorithm 1 / M12 objective
src/preprocessing.py            shared utils + per-fold imputation
src/full_diagnostic_battery.py  the full corrected battery (all 5 parts)
src/field_pipeline.py           primary corpus runner
src/ibm_pipeline.py             validation corpus runner
requirements.txt                exact pinned versions
```

## Reproduce

```bash
pip install -r requirements.txt
python src/ibm_pipeline.py                       # public data, auto-downloads
python src/field_pipeline.py --data <your.xlsx>  # restricted field data
```

A fixed seed (42) governs partitioning, SMOTE, and model stochasticity.
A supplementary decomposition shows fold-partition variance exceeds
model-seed variance ~550x, so the single-seed design understates total
variance only marginally.

## Data

The field corpus is restricted (Methods M5/M21). The IBM HR Analytics
validation corpus is public and downloads automatically.

## Licence

MIT — see LICENSE. IBM HR Analytics retains its own original licence.
