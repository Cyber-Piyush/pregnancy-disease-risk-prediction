# Pregnancy Disease Risk Prediction

Educational prototype that predicts a pregnancy-related disease-risk category from health inputs using a Random Forest classifier.

> **Disclaimer — read first:** This is an educational prototype built on **synthetic, rule-generated data**. It is **not** a medical diagnostic tool, has not been validated on real patient data, and must not be used for medical decisions.

## What it does

- Generates 1,000 synthetic patients itself (seed 42): age, BMI, glucose, systolic/diastolic blood pressure, haemoglobin.
- Assigns the target with fixed if/else rules, e.g. very high blood pressure + high BMI + age over 30 → `preeclampsia_high`.
- Trains a Random Forest classifier (`n_estimators=100`, 80/20 stratified split) on those 6 features.
- Prints 4 sample predictions (Alice, Brenda, Clara, Diana).

## Verified result

Run 4 November 2026 in a clean environment: **test accuracy 96.50%**.

This number is **not** evidence of clinical performance — the code itself notes high accuracy is expected because the data is fake and rule-based. The target distribution is also heavily imbalanced: 552 of 1,000 samples are `hypertension_high`, only 16 are `anaemia_low`.

Target distribution (1,000 samples):

| Category | Count |
|---|---|
| hypertension_high | 552 |
| preeclampsia_high | 111 |
| diabetes_moderate | 79 |
| anaemia_moderate | 76 |
| healthy_low | 75 |
| diabetes_high | 49 |
| hypertension_moderate | 42 |
| anaemia_low | 16 |

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_rule_based.py
```

## Tech stack

Python, pandas, NumPy, scikit-learn (Random Forest).

## Project structure

```
pregnancy-disease-risk-prediction/
├── train_rule_based.py
├── requirements.txt
├── .gitignore
└── TODO.md
```

## Limitations

- Synthetic / rule-based data only; no real clinical validation.
- No baseline comparison (e.g. logistic regression) was run.
- Class imbalance (see distribution above).
- No web interface in this version — the earlier Flask frontend depended on a separate CSV-based pipeline whose dataset and model files are not available, so it was deliberately left out rather than published broken.
- No tests or screenshots yet.

## Notes on the earlier version

An earlier CSV-based version of this project (two Random Forests, 18 features, Flask backend) is **not** included here, because its dataset (`synthetic_pregnancy_disease_data_with_normals.csv`) and trained model files are not available. Publishing it without them would have left a repository that cannot run. If those files are recovered later, that version can be added back as a separate, clearly labelled branch.
