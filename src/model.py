"""
LightGBM training, cross-validation, and Optuna hyperparameter tuning
for the Home Credit Default Risk pipeline.
"""

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Silence Optuna's per-trial log output unless the caller wants it
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Default LightGBM parameters (used for baseline and full-feature runs)
DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "random_state": 42,
}

CV_SPLITS = 5
CV_RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50


def run_cv(X, y, params: dict | None = None, n_splits: int = CV_SPLITS) -> tuple[list[float], lgb.LGBMClassifier]:
    """
    Run stratified k-fold cross-validation with LightGBM.

    Parameters
    ----------
    X        : Feature matrix (pandas DataFrame or numpy array).
    y        : Binary target vector.
    params   : LGBMClassifier kwargs. Defaults to DEFAULT_PARAMS.
    n_splits : Number of CV folds.

    Returns
    -------
    scores : List of per-fold ROC-AUC scores.
    model  : The model trained on the last fold (useful for feature importance).
    """
    if params is None:
        params = DEFAULT_PARAMS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
    scores = []
    model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(100),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)
        scores.append(auc)
        print(f"  Fold {fold + 1} AUC: {auc:.4f}")

    print(f"\n  Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores, model


def _make_objective(X, y, n_splits: int = 3):
    """Return an Optuna objective function closed over X and y."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 1000,
            "random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
        fold_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(-1),
                ],
            )

            preds = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, preds))

        return float(np.mean(fold_scores))

    return objective


def tune(X, y, n_trials: int = 20) -> dict:
    """
    Run an Optuna hyperparameter search over key LightGBM parameters.

    Parameters
    ----------
    X        : Feature matrix.
    y        : Binary target vector.
    n_trials : Number of Optuna trials.

    Returns
    -------
    best_params : Dict of best hyperparameters (including n_estimators and
                  random_state, ready to pass directly to LGBMClassifier).
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(_make_objective(X, y), n_trials=n_trials)

    print(f"\n  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best_params = study.best_params
    best_params["n_estimators"] = 1000
    best_params["random_state"] = 42
    return best_params


def train_final_model(X, y, params: dict | None = None) -> lgb.LGBMClassifier:
    """
    Train a single LightGBM model on the full training set.

    Parameters
    ----------
    X      : Full feature matrix (no val split).
    y      : Full target vector.
    params : LGBMClassifier kwargs. Defaults to DEFAULT_PARAMS.

    Returns
    -------
    Fitted LGBMClassifier.
    """
    if params is None:
        params = DEFAULT_PARAMS

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(100)])
    return model


def predict(model: lgb.LGBMClassifier, X_test) -> np.ndarray:
    """Return predicted default probabilities for the test set."""
    return model.predict_proba(X_test)[:, 1]