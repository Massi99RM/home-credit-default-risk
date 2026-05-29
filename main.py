"""
Home Credit Default Risk — end-to-end pipeline.

Usage
-----
  # Full run with default params:
  python main.py

  # Full run + Optuna tuning:
  python main.py --tune

  # Custom data / output directories:
  python main.py --data-dir path/to/data/raw --output-dir path/to/outputs

  # Tune with more trials:
  python main.py --tune --n-trials 50
"""

import argparse
import os

import pandas as pd

from src.features import build_features
from src.model import run_cv, tune, train_final_model, predict, DEFAULT_PARAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Credit Default Risk pipeline")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory containing raw CSV files (default: data/raw)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for submission CSV (default: outputs)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search before final training")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20, only used with --tune)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load main training table
    # ------------------------------------------------------------------
    print("\n=== Loading training data ===")
    train = pd.read_csv(f"{args.data_dir}/application_train.csv")
    print(f"  application_train shape: {train.shape}")

    # ------------------------------------------------------------------
    # 2. Baseline: CV on the main table only
    # ------------------------------------------------------------------
    print("\n=== Baseline CV (main table only) ===")
    train_base = train.copy()
    for col in train_base.select_dtypes("object").columns:
        train_base[col] = pd.Categorical(train_base[col]).codes

    X_base = train_base.drop(["SK_ID_CURR", "TARGET"], axis=1)
    y_base = train_base["TARGET"]
    run_cv(X_base, y_base)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Building features ===")
    train_full = build_features(train, args.data_dir)
    print(f"  Enriched training shape: {train_full.shape}")

    X = train_full.drop(["SK_ID_CURR", "TARGET"], axis=1)
    y = train_full["TARGET"]

    # ------------------------------------------------------------------
    # 4. Full-feature CV
    # ------------------------------------------------------------------
    print("\n=== Full-feature CV ===")
    run_cv(X, y)

    # ------------------------------------------------------------------
    # 5. Optional Optuna tuning
    # ------------------------------------------------------------------
    final_params = DEFAULT_PARAMS
    if args.tune:
        print(f"\n=== Optuna tuning ({args.n_trials} trials) ===")
        final_params = tune(X, y, n_trials=args.n_trials)

    # ------------------------------------------------------------------
    # 6. Train final model on all training data
    # ------------------------------------------------------------------
    print("\n=== Training final model ===")
    final_model = train_final_model(X, y, params=final_params)

    # ------------------------------------------------------------------
    # 7. Prepare test set and predict
    # ------------------------------------------------------------------
    print("\n=== Preparing test data ===")
    test = pd.read_csv(f"{args.data_dir}/application_test.csv")
    test_ids = test["SK_ID_CURR"]
    print(f"  application_test shape: {test.shape}")

    test_full = build_features(test, args.data_dir)
    X_test = test_full.drop(["SK_ID_CURR"], axis=1)

    # Align columns: drop any test columns not in train, add missing as NaN.
    # NaN lets LightGBM route samples using the direction learned at training
    # time; filling with 0 would assign a real value that may land on the
    # wrong side of a split.
    missing_cols = set(X.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = float("nan")
    X_test = X_test[X.columns]

    preds = predict(final_model, X_test)

    # ------------------------------------------------------------------
    # 8. Save submission
    # ------------------------------------------------------------------
    submission_path = f"{args.output_dir}/submission.csv"
    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": preds})
    submission.to_csv(submission_path, index=False)
    print(f"\n=== Done. Submission saved to {submission_path} ===")
    print(submission.head())


if __name__ == "__main__":
    main()