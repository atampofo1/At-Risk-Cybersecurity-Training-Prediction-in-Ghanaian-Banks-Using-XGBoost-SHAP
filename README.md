# Explainable Learning Analytics for At-Risk Cybersecurity Training Prediction in Ghanaian Banks Using XGBoost-SHAP

Reference implementation for the Methods and Results sections (Student:
Alex Tawiah Ampofo, 22527933; Supervisor: Dr. Eric Opoku Osei).

## Headline finding (Results R10)

Under a corrected, leakage-free nested cross-validation with a matched
inner-fold tuning budget across all comparators, the proposed focal-loss
substitution does **not** produce a statistically significant improvement
over a properly tuned baseline on either corpus (field: dAUC -0.0054,
p = 0.890; validation: dAUC +0.0067, p = 0.095). Two re-engineering
attempts (an interaction feature; a class-weighted loss) also fail once
their own selection-on-test defects are corrected. This is reported as a
diagnosed negative result, not concealed.

## Corrections applied in this revision (verification pass K1-K5)

- **K1** Parts 2-5 of the diagnostic battery use the full unselected
  candidate-feature set (no selection outside a fold; nothing to leak).
- **K2/K2b/K2c** Every tuned arm (baseline, proposed, RF, LR, SVM)
  receives an equal inner-fold trial budget; LR and SVM are now genuinely
  tuned rather than left at defaults. Search-space dimensionality still
  is equal across arms: five genuine hyperparameters each (Methods M13,
  Table 3b). Per-fold selected values for every arm are emitted under
  `tuned_hyperparameters_per_fold` in the battery JSON.
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
model-seed variance ~513x (field) / ~824x (validation), so the single-seed design understates total
variance only marginally.

## Data

The field corpus is restricted (Methods M5/M21). The IBM HR Analytics
validation corpus is public and downloads automatically.

## Licence

MIT — see LICENSE. IBM HR Analytics retains its own original licence.


## Round 3 corrections (18 Sep 2026)
- Ablation row 2 (focal-loss-only) is now fitted with the baseline's per-fold
  Optuna-tuned tree hyperparameters, so all three ablation rows share one
  hyperparameter regime and M16/R5 can state constancy accurately (Q10 close).
- Manuscript numeric-integrity fixes: M2 feature count corrected to d=9 (field)
  / d=12 (validation); table/figure renumbering; Accuracy column and trivial-
  classifier floors added; 95% CIs on both corpora; Table 7 stale numbers
  refreshed; per-fold test-n reported.
- All four Q19 negative-result-honesty sentences written (calibration/Brier
  mechanism, error-count identity, M15 cost resolution, two n=107 split
  distinction).
- M21 now cites the live repository URL.

## Round 4 corrections (5 Nov 2026, against the Q1 Methods marking report)
- Battery now records every arm's selected hyperparameters for each of the
  15 outer folds (`tuned_hyperparameters_per_fold`); no metric changed
  (re-run reproduced all Table 4/5/8 values exactly).
- README headline numbers corrected to the verified Procedure A results.
- IBM file integrity: SHA-256 a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659
  (1,470 rows, 35 columns, 237 attrition = Yes).
- Environment used: Ubuntu 24.04, Python 3.13, 2 vCPU Intel Xeon 2.80 GHz,
  7 GB RAM; library versions pinned in requirements.txt.
- Feature-selection cap made explicit (`N_SELECT_CAP = 12` in
  `src/full_diagnostic_battery.py`). The earlier "8 if 10 or fewer candidates"
  branch could never trigger for either corpus (11 and 16 candidates), so it was
  removed; a full rerun reproduced every result field exactly (d = 9 field,
  d = 12 validation).
