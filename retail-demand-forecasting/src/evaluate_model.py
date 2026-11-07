"""
evaluate_model.py
-----------------
STEP 4 of the pipeline: explain HOW GOOD the predictions are.

Three things happen here:
    1. Test-set predictions of every model are compared
       -> outputs/test_predictions.csv (used by the dashboard too).
    2. Errors are broken down PER PRODUCT CATEGORY
       -> outputs/category_errors.csv
       This answers a real business question: "which products do we
       forecast badly, and where should we keep extra safety stock?"
    3. Charts are drawn:
       -> 06_actual_vs_predicted.png
       -> 07_model_comparison.png
       -> 09_category_wise_errors.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config

sns.set_theme(style="whitegrid")


def build_test_predictions_table(test_df: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """
    Build a tidy table with the actual sales and each model's prediction
    for every row of the test set.
    """
    table = test_df[["date", "store_id", "product_id", "product_category",
                     config.TARGET_COL]].copy()
    table = table.rename(columns={config.TARGET_COL: "actual_sales"})

    for model_name, info in predictions.items():
        column = model_name.lower().replace(" ", "_") + "_prediction"
        table[column] = np.clip(info["predictions"], 0, None)

    return table


def category_error_table(test_df: pd.DataFrame, best_predictions: np.ndarray) -> pd.DataFrame:
    """
    Compute MAE / RMSE / record count for each product category.

    Only the SELECTED model is used here - we want to know where the model we
    actually ship performs badly.
    """
    work = test_df[["product_category", config.TARGET_COL]].copy()
    work["prediction"] = np.clip(best_predictions, 0, None)

    rows = []
    for category, group in work.groupby("product_category"):
        rows.append(
            {
                "product_category": category,
                "MAE": round(mean_absolute_error(group[config.TARGET_COL], group["prediction"]), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(
                    group[config.TARGET_COL], group["prediction"]))), 3),
                "test_records": len(group),
                # MAPE: average percentage error, easier to communicate
                "MAPE_percent": round(float(np.mean(
                    np.abs(group[config.TARGET_COL] - group["prediction"])
                    / np.maximum(group[config.TARGET_COL], 1)) * 100), 2),
            }
        )

    table = pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)
    return table


def save_category_errors(table: pd.DataFrame) -> None:
    """Save the category-wise error table."""
    config.ensure_directories()
    table.to_csv(config.CATEGORY_ERROR_FILE, index=False, encoding="utf-8")
    print(f"Category error table saved to: {config.CATEGORY_ERROR_FILE}")


def save_test_predictions(table: pd.DataFrame) -> None:
    """Save the test-set predictions (used by the Streamlit dashboard)."""
    config.ensure_directories()
    table.to_csv(config.TEST_PREDICTIONS_FILE, index=False, encoding="utf-8")
    print(f"Test predictions saved to: {config.TEST_PREDICTIONS_FILE}")


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def plot_actual_vs_predicted(test_predictions: pd.DataFrame, model_name: str) -> None:
    """
    Two panels:
      left  - actual vs predicted for ONE store-product series (readable)
      right - scatter of all test rows with the perfect-prediction line
    """
    prediction_col = model_name.lower().replace(" ", "_") + "_prediction"

    # Pick the store-product pair with the most test rows -> a smooth series
    series_key = (
        test_predictions.groupby(["store_id", "product_id"]).size().idxmax()
    )
    subset = test_predictions[
        (test_predictions["store_id"] == series_key[0])
        & (test_predictions["product_id"] == series_key[1])
    ].sort_values("date")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(subset["date"], subset["actual_sales"], label="Actual sales",
                 color="#1d3557", linewidth=1.8)
    axes[0].plot(subset["date"], subset[prediction_col], label=f"Predicted ({model_name})",
                 color="#e63946", linewidth=1.5, linestyle="--")
    axes[0].set_title(f"Actual vs Predicted - Store {series_key[0]}, Product {series_key[1]}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Units sold")
    axes[0].legend()
    for label in axes[0].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    axes[1].scatter(test_predictions["actual_sales"], test_predictions[prediction_col],
                    alpha=0.15, s=8, color="#457b9d")
    limit = int(max(test_predictions["actual_sales"].max(), test_predictions[prediction_col].max()))
    axes[1].plot([0, limit], [0, limit], color="#e63946", linestyle="--",
                 label="Perfect prediction")
    axes[1].set_title("All Test Records: Actual vs Predicted")
    axes[1].set_xlabel("Actual units sold")
    axes[1].set_ylabel("Predicted units sold")
    axes[1].legend()

    fig.suptitle(f"Actual vs Predicted Sales ({model_name})", fontsize=15)
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(config.CHART_ACTUAL_VS_PREDICTED, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {config.CHART_ACTUAL_VS_PREDICTED.name}")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar charts comparing MAE, RMSE and R2 for every model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = sns.color_palette("viridis", len(results_df))

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.bar(results_df["model"], results_df[metric], color=colors)
        ax.set_title(f"{metric} by Model (test set)")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric + (" (lower is better)" if metric != "R2" else " (higher is better)"))
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        # Write the value on top of each bar so the numbers are readable
        for bar, value in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}" if metric != "R2" else f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=15)
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(config.CHART_MODEL_COMPARISON, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {config.CHART_MODEL_COMPARISON.name}")


def plot_category_errors(category_table: pd.DataFrame) -> None:
    """Bar chart of MAE and RMSE per product category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=category_table, x="product_category", y="MAE", ax=axes[0],
                hue="product_category", palette="rocket", legend=False)
    axes[0].set_title("Mean Absolute Error by Product Category")
    axes[0].set_xlabel("Product category")
    axes[0].set_ylabel("MAE (units)")
    for i, value in enumerate(category_table["MAE"]):
        axes[0].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    sns.barplot(data=category_table, x="product_category", y="RMSE", ax=axes[1],
                hue="product_category", palette="mako", legend=False)
    axes[1].set_title("RMSE by Product Category")
    axes[1].set_xlabel("Product category")
    axes[1].set_ylabel("RMSE (units)")
    for i, value in enumerate(category_table["RMSE"]):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Category-wise Prediction Errors (selected model)", fontsize=15)
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(config.CHART_CATEGORY_ERRORS, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {config.CHART_CATEGORY_ERRORS.name}")


def run_evaluation(results_df, trained_models, best_model_name,
                   train_df, test_df) -> pd.DataFrame:
    """
    Run the whole evaluation step and return the category error table.
    'trained_models' is the dict returned by train_model.train_and_compare().
    """
    print("\n--- EVALUATING THE SELECTED MODEL ---")

    best_predictions = trained_models[best_model_name]["predictions"]
    test_predictions = build_test_predictions_table(test_df, trained_models)
    save_test_predictions(test_predictions)

    category_table = category_error_table(test_df, best_predictions)
    print("\nCategory-wise errors (selected model):")
    print(category_table.to_string(index=False))
    save_category_errors(category_table)

    print("\nCreating evaluation charts...")
    plot_actual_vs_predicted(test_predictions, best_model_name)
    plot_model_comparison(results_df)
    plot_category_errors(category_table)

    # Explain the result in plain words
    worst = category_table.iloc[0]
    best = category_table.iloc[-1]
    print(f"\nHighest error category : {worst['product_category']} (MAE {worst['MAE']:.2f})")
    print(f"Lowest error category  : {best['product_category']} (MAE {best['MAE']:.2f})")

    return category_table
