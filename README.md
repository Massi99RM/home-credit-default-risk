# Home Credit Default Risk

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A full tabular ML pipeline for predicting loan default probability, built on the [Home Credit Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk). The focus is on relational feature engineering across multiple tables, proper cross-validation, and hyperparameter tuning with Optuna.

## Overview

Many people who struggle to get loans lack a conventional credit history. Home Credit uses alternative data — installment payments, credit bureau records, previous loan behavior — to assess repayment ability. The task is to predict, for each applicant, the probability they will default on their loan.

This project covers the full pipeline a real ML role would require: joining relational tables, aggregating historical data, building a validated model, and tuning it systematically.

### Key Results

| Metric | Value |
|--------|-------|
| Baseline ROC-AUC (main table only) | 0.7570  |
| Full model ROC-AUC (all tables) | TBD |
| CV Strategy | 5-Fold Stratified |
| Tuning | Optuna |

## How It Works

### Problem

Predict `TARGET` (1 = defaulted, 0 = repaid) for each `SK_ID_CURR` in the test set. Evaluated by ROC-AUC: how well the model ranks defaulters above non-defaulters across all thresholds.

### Data Structure

The dataset consists of one main table and six auxiliary tables linked by `SK_ID_CURR`:

```
application_train.csv        ← one row per applicant, contains TARGET
        │
        ├── bureau.csv                  ← previous credits at other banks
        │       └── bureau_balance.csv  ← monthly history of those credits
        ├── previous_application.csv    ← previous Home Credit applications
        ├── POS_CASH_balance.csv        ← monthly POS and cash loan snapshots
        ├── credit_card_balance.csv     ← monthly credit card snapshots
        └── installments_payments.csv   ← repayment history
```

Each auxiliary table has many rows per applicant. They must be aggregated (mean, max, min, count, std) before joining to the main table.

### Pipeline

1. **EDA** — class imbalance, missing values, feature distributions on the main table
2. **Baseline** — LightGBM with 5-fold stratified CV on the main table only
3. **Feature engineering** — aggregate all auxiliary tables, join to main table
4. **Full model** — LightGBM on the enriched feature set, same CV strategy
5. **Hyperparameter tuning** — Optuna search over key LightGBM parameters
6. **Submission** — predict probabilities on the test set, submit to Kaggle


## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle.

- ~307,000 loan applications in the training set
- ~48,000 in the test set
- 8% default rate (class imbalance)
- 7 relational tables, ~120 raw features, 300+ after aggregation

## License

MIT