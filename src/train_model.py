"""
train_model.py
--------------
STEP 3 of the pipeline: train, tune and compare three regression models.

MODELS
    1. Linear Regression            - simple baseline, easy to explain
    2. Random Forest Regressor      - default settings
    3. Tuned Random Forest          - small GridSearchCV over a tiny grid

METRICS (computed on the TEST set, i.e. the future)
    MAE  - average absolute error in units. "We are off by ~X units per row."
           Very easy to explain to a business audience.
    RMSE - like MAE but punishes big mistakes more. Good for spotting days
           where we badly under-stocked.
    R2   - how much of the variation in sales the model explains (1 = perfect).

MODEL SELECTION
    We pick the model with the LOWEST RMSE on the test set, because in retail
    a big under-forecast causes a stock-out (lost sales) and a big over-forecast
    causes dead stock. RMSE punishes those large misses, so it matches the
    business cost better than MAE alone.
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def calculate_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R2 in one dictionary."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
def build_baseline_model() -> LinearRegression:
    """Model 1: plain Linear Regression (the baseline to beat)."""
    return LinearRegression()


def build_random_forest_model() -> RandomForestRegressor:
    """Model 2: Random Forest with sensible defaults."""
    return RandomForestRegressor(
        n_estimators=200,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,   # use all CPU cores
    )


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Model 3: tune a Random Forest with a SMALL grid.

    TimeSeriesSplit is used because our data is a time series: each
    cross-validation fold trains on earlier dates and validates on later dates,
    exactly like the final test. A normal KFold would leak the future.
    """
    print("\nTuning Random Forest (small grid, TimeSeriesSplit = 3)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [12, None],
        "min_samples_leaf": [1, 3],
    }
    # n_jobs=1 here: GridSearchCV already runs folds in parallel, and nesting
    # parallel jobs inside parallel jobs is slower, not faster.
    base_model = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation RMSE: {-search.best_score_:.3f}")

    return search.best_estimator_


# ----------------------------------------------------------------------
# Train + compare everything
# ----------------------------------------------------------------------
def train_and_compare(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all three models, evaluate them on the test set and return:
        results_df, trained_models_dict, best_model_name, best_model
    """
    print("\n--- TRAINING MODELS ---")

    models = {}
    models["Linear Regression"] = build_baseline_model()
    models["Random Forest"] = build_random_forest_model()

    results = []
    trained = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, 0, None)   # sales cannot be negative
        metrics = calculate_metrics(y_test, predictions)
        trained[name] = {"model": model, "predictions": predictions, "metrics": metrics}
        results.append({"model": name, **metrics, "train_seconds": round(time.time() - start, 2)})
        print(f"{name:<20} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")

    # ---- Tuned Random Forest ----
    start = time.time()
    tuned_rf = tune_random_forest(X_train, y_train)
    tuned_rf.fit(X_train, y_train)
    predictions = np.clip(tuned_rf.predict(X_test), 0, None)
    metrics = calculate_metrics(y_test, predictions)
    trained["Tuned Random Forest"] = {"model": tuned_rf, "predictions": predictions, "metrics": metrics}
    results.append({"model": "Tuned Random Forest", **metrics,
                    "train_seconds": round(time.time() - start, 2)})
    print(f"{'Tuned Random Forest':<20} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)[["model", "MAE", "RMSE", "R2", "train_seconds"]]
    results_df["MAE"] = results_df["MAE"].round(3)
    results_df["RMSE"] = results_df["RMSE"].round(3)
    results_df["R2"] = results_df["R2"].round(4)

    # ---- Select the winner: lowest test RMSE ----
    best_model_name = results_df.loc[results_df["RMSE"].idxmin(), "model"]
    best_model = trained[best_model_name]["model"]
    print(f"\nSelected model (lowest test RMSE): {best_model_name}")

    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    return results_df, trained, best_model_name, best_model


def save_model_artifacts(best_model, best_model_name: str, results_df: pd.DataFrame,
                         train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save the trained model + the feature list + metadata with joblib."""
    config.ensure_directories()

    # compress=3 keeps the file small (a Random Forest is mostly redundant trees)
    # and loads back in the same way with joblib.load().
    joblib.dump(best_model, config.MODEL_FILE, compress=3)
    joblib.dump(config.FEATURE_COLUMNS, config.FEATURES_FILE)

    metadata = {
        "model_name": best_model_name,
        "feature_columns": config.FEATURE_COLUMNS,
        "test_metrics": results_df.iloc[0].to_dict(),
        "train_start": str(train_df[config.DATE_COL].min().date()),
        "train_end": str(train_df[config.DATE_COL].max().date()),
        "test_start": str(test_df[config.DATE_COL].min().date()),
        "test_end": str(test_df[config.DATE_COL].max().date()),
        "random_state": config.RANDOM_STATE,
    }
    joblib.dump(metadata, config.METADATA_FILE)

    print(f"Model saved to      : {config.MODEL_FILE}")
    print(f"Feature list saved  : {config.FEATURES_FILE}")
    print(f"Metadata saved      : {config.METADATA_FILE}")


def save_evaluation_report(results_df: pd.DataFrame) -> None:
    """Save the model comparison table to outputs/evaluation_report.csv."""
    config.ensure_directories()
    results_df.to_csv(config.EVALUATION_FILE, index=False, encoding="utf-8")
    print(f"Evaluation report saved to: {config.EVALUATION_FILE}")
