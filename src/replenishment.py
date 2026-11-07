"""
replenishment.py
----------------
STEP 6 of the pipeline: turn forecasts into ORDER decisions.

THE BUSINESS QUESTION
    "I know roughly how much I will sell next week. How much should I order
     today so I do not run out of stock while the supplier is delivering?"

THE FORMULAS (kept deliberately simple)
    expected_demand  = sum of predicted demand for the next `lead_time` days
                       (the days the supplier needs to deliver)

    safety_stock     = 1.65 * std(daily demand, last 30 days) * sqrt(lead_time)
                       1.65 is the z-value for ~95% service level: it covers
                       normal day-to-day ups and downs. sqrt(lead_time) makes
                       the buffer grow with the length of the delivery period,
                       but slower than the period itself (risk pooling).

    reorder_quantity = max(0, expected_demand + safety_stock - current_stock)
                       i.e. order the gap between what we will sell (+ buffer)
                       and what we already have.

ASSUMPTIONS (state these out loud in a review)
    - Lead time is 3 days (config.DEFAULT_LEAD_TIME_DAYS) because the dataset
      has no supplier information. A real system reads it per supplier.
    - Current stock is the LAST recorded stock_available for that store-product.
    - Demand variability is measured on the last 30 REAL sales days.
    - We assume no orders are already in transit and no minimum order quantity.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


def calculate_reorder_recommendations(
    forecasts: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    lead_time_days: int = config.DEFAULT_LEAD_TIME_DAYS,
    safety_z: float = config.SAFETY_STOCK_Z,
) -> pd.DataFrame:
    """
    Build the reorder table for every store-product pair.

    Returns a DataFrame with one row per store-product:
        store_id, product_id, product_category, current_stock,
        expected_demand, safety_stock, recommended_reorder_quantity, ...
    """
    print(f"\n--- REPLENISHMENT (lead time = {lead_time_days} days) ---")

    # ---- 1. Expected demand during the lead time ----------------------
    # Only the first `lead_time_days` forecast days matter: goods ordered today
    # arrive after the lead time, so we must cover exactly those days.
    lead_time_window = (
        forecasts.sort_values("date")
        .groupby(config.GROUP_COLS)
        .head(lead_time_days)                       # first N forecast days per series
        .groupby(config.GROUP_COLS)["predicted_demand"]
        .sum()
        .rename("expected_demand")
    )

    # ---- 2. Demand variability from the last 30 real days --------------
    recent = (
        cleaned_df.sort_values(config.DATE_COL)
        .groupby(config.GROUP_COLS)
        .tail(config.DEMAND_STD_WINDOW)
    )
    demand_std = (
        recent.groupby(config.GROUP_COLS)[config.TARGET_COL]
        .std()                       # sample standard deviation (ddof=1)
        .fillna(0.0)
        .rename("demand_std")
    )

    # ---- 3. Current stock = latest recorded stock ----------------------
    latest_stock = (
        cleaned_df.sort_values(config.DATE_COL)
        .groupby(config.GROUP_COLS)
        .agg(
            current_stock=("stock_available", "last"),
            product_category=("product_category", "last"),
        )
    )

    # ---- 4. Put the pieces together ------------------------------------
    table = latest_stock.join(lead_time_window).join(demand_std).reset_index()
    table["expected_demand"] = table["expected_demand"].fillna(0)
    table["demand_std"] = table["demand_std"].fillna(0)

    # Safety stock formula (see explanation at the top of this file)
    table["safety_stock"] = safety_z * table["demand_std"] * np.sqrt(lead_time_days)

    # Reorder quantity: never negative
    table["recommended_reorder_quantity"] = np.maximum(
        0.0,
        table["expected_demand"] + table["safety_stock"] - table["current_stock"],
    )

    # Round to whole units and round UP, so we never order half a unit short
    table["expected_demand"] = table["expected_demand"].round(2)
    table["safety_stock"] = table["safety_stock"].round(2)
    table["recommended_reorder_quantity"] = np.ceil(
        table["recommended_reorder_quantity"]
    ).astype(int)

    table["lead_time_days"] = lead_time_days
    table["needs_reorder"] = (table["recommended_reorder_quantity"] > 0).astype(int)

    # Nice, readable column order for the CSV / dashboard
    table = table[
        [
            "store_id",
            "product_id",
            "product_category",
            "current_stock",
            "expected_demand",
            "safety_stock",
            "recommended_reorder_quantity",
            "lead_time_days",
            "needs_reorder",
        ]
    ].sort_values("recommended_reorder_quantity", ascending=False).reset_index(drop=True)

    n_reorder = int(table["needs_reorder"].sum())
    print(f"Store-products analysed      : {len(table)}")
    print(f"Products needing a reorder   : {n_reorder} "
          f"({n_reorder / len(table) * 100:.1f}%)")
    print(f"Total units recommended      : {table['recommended_reorder_quantity'].sum():,}")

    return table


def save_reorder_recommendations(table: pd.DataFrame) -> None:
    """Save outputs/reorder_recommendations.csv"""
    config.ensure_directories()
    table.to_csv(config.REORDER_FILE, index=False, encoding="utf-8")
    print(f"Reorder recommendations saved to: {config.REORDER_FILE}")


def explain_with_example(table: pd.DataFrame) -> None:
    """
    Print a worked example of one row, so the calculation is easy to explain
    in a presentation ("this is how the number was produced").
    """
    example = table.iloc[0]
    print("\nWorked example (the row with the largest reorder quantity):")
    print(f"  Store {example['store_id']}, Product {example['product_id']} "
          f"({example['product_category']})")
    print(f"  Current stock                    : {example['current_stock']:.0f} units")
    print(f"  Expected demand during lead time : {example['expected_demand']:.2f} units")
    print(f"  Safety stock                     : {example['safety_stock']:.2f} units")
    print(f"  Reorder = max(0, {example['expected_demand']:.2f} + "
          f"{example['safety_stock']:.2f} - {example['current_stock']:.0f}) "
          f"= {example['recommended_reorder_quantity']} units")


def plot_stock_vs_reorder(table: pd.DataFrame, top_n: int = 15) -> None:
    """
    Chart #10: current stock vs recommended reorder quantity.

    Left  - the top N store-products that need the most units.
    Right - how many store-products need a reorder at all.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top = table.head(top_n).copy()
    top["label"] = top["store_id"] + " / " + top["product_id"]
    positions = np.arange(len(top))
    width = 0.4

    axes[0].barh(positions - width / 2, top["current_stock"], width,
                 label="Current stock", color="#8ecae6")
    axes[0].barh(positions + width / 2, top["recommended_reorder_quantity"], width,
                 label="Recommended reorder", color="#e63946")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(top["label"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Top {top_n} Items: Current Stock vs Recommended Reorder")
    axes[0].set_xlabel("Units")
    axes[0].legend()

    counts = table["needs_reorder"].map({1: "Reorder needed", 0: "Stock sufficient"}).value_counts()
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
                colors=["#e63946", "#81b29a"])
    axes[1].set_title("Store-Products Requiring a Reorder")

    fig.suptitle("Replenishment Recommendations", fontsize=15)
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(config.CHART_STOCK_VS_REORDER, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {config.CHART_STOCK_VS_REORDER.name}")
