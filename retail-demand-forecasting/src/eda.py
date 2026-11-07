"""
eda.py
------
STEP 2b: Exploratory Data Analysis (EDA) - charts and summary tables.

The functions here are used by BOTH the notebook (notebooks/eda.ipynb) and
main.py, so the notebook shows exactly the same numbers as the pipeline.

Every chart is saved as a PNG in outputs/charts/ and also returned as a
Figure object, so the notebook can display it inline.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend: writes PNG files, no pop-up windows

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

# A consistent, readable look for all charts
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


def _save(fig, path) -> None:
    """Small helper: save a figure with tight layout and close it."""
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {path.name}")


# ----------------------------------------------------------------------
# 1. Dataset overview as plain text
# ----------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a short overview of the dataset."""
    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"Date range: {df[config.DATE_COL].min().date()} -> {df[config.DATE_COL].max().date()}")
    print(f"Stores: {df['store_id'].nunique()}   Products: {df['product_id'].nunique()}   "
          f"Categories: {df['product_category'].nunique()}")

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(),
        }
    )
    print("\nColumn information:")
    print(summary.to_string())
    print("\nNumeric summary:")
    print(df[["sales_quantity", "unit_price", "stock_available", "promotion"]]
          .describe().round(2).to_string())
    return summary


# ----------------------------------------------------------------------
# 2. Sales distribution
# ----------------------------------------------------------------------
def plot_sales_distribution(df: pd.DataFrame) -> None:
    """Histogram + box plot of daily sales quantities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[config.TARGET_COL], bins=45, kde=True, ax=axes[0], color="#2a6f97")
    axes[0].set_title("Distribution of Daily Sales Quantity")
    axes[0].set_xlabel("Sales quantity (units)")
    axes[0].set_ylabel("Number of records")

    sns.boxplot(x=df[config.TARGET_COL], ax=axes[1], color="#61a5c2")
    axes[1].set_title("Box Plot of Daily Sales Quantity")
    axes[1].set_xlabel("Sales quantity (units)")

    fig.suptitle("Sales Distribution", fontsize=15)
    _save(fig, config.CHARTS_DIR / "00_sales_distribution.png")


# ----------------------------------------------------------------------
# 3. Daily sales trend  (chart #1 required by the brief)
# ----------------------------------------------------------------------
def plot_daily_trend(df: pd.DataFrame) -> None:
    """Total sales per day, with a 30-day rolling average to show the trend."""
    daily = df.groupby(config.DATE_COL)[config.TARGET_COL].sum().reset_index()
    daily["rolling_30"] = daily[config.TARGET_COL].rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily[config.DATE_COL], daily[config.TARGET_COL],
            color="#8ecae6", linewidth=0.8, label="Daily total sales")
    ax.plot(daily[config.DATE_COL], daily["rolling_30"],
            color="#d62828", linewidth=2.2, label="30-day rolling average")
    ax.set_title("Daily Sales Trend (All Stores)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total units sold")
    ax.legend(loc="upper left")
    _save(fig, config.CHART_DAILY_TREND)


# ----------------------------------------------------------------------
# 4. Monthly sales trend  (chart #2)
# ----------------------------------------------------------------------
def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Total sales per month as a bar chart, plus average sales per day line."""
    monthly = (
        df.assign(month_start=df[config.DATE_COL].dt.to_period("M").dt.to_timestamp())
        .groupby("month_start")[config.TARGET_COL]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=monthly, x="month_start", y=config.TARGET_COL,
                ax=ax, color="#457b9d")
    ax.set_title("Monthly Sales Trend (All Stores)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total units sold")
    # rotate x labels so the dates are readable
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    _save(fig, config.CHART_MONTHLY_TREND)


# ----------------------------------------------------------------------
# 5. Category-wise sales  (chart #3)
# ----------------------------------------------------------------------
def plot_category_sales(df: pd.DataFrame) -> None:
    """Bar chart + share of total for each product category."""
    category_sales = (
        df.groupby("product_category")[config.TARGET_COL].sum().sort_values(ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=category_sales.index, y=category_sales.values, ax=axes[0],
                hue=category_sales.index, palette="viridis", legend=False)
    axes[0].set_title("Total Sales by Product Category")
    axes[0].set_xlabel("Product category")
    axes[0].set_ylabel("Total units sold")
    for i, value in enumerate(category_sales.values):
        axes[0].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].pie(category_sales.values, labels=category_sales.index, autopct="%1.1f%%",
                startangle=90, colors=sns.color_palette("viridis", len(category_sales)))
    axes[1].set_title("Share of Total Sales")
    _save(fig, config.CHART_CATEGORY_SALES)


# ----------------------------------------------------------------------
# 6. Store-wise sales  (extra, used by the dashboard)
# ----------------------------------------------------------------------
def plot_store_sales(df: pd.DataFrame) -> None:
    """Bar chart of total sales per store."""
    store_sales = df.groupby("store_id")[config.TARGET_COL].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=store_sales.index, y=store_sales.values, ax=ax,
                hue=store_sales.index, palette="mako", legend=False)
    ax.set_title("Total Sales by Store")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total units sold")
    for i, value in enumerate(store_sales.values):
        ax.text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    _save(fig, config.CHART_STORE_SALES)


# ----------------------------------------------------------------------
# 7. Promotion vs non-promotion  (chart #5)
# ----------------------------------------------------------------------
def plot_promotion_effect(df: pd.DataFrame) -> None:
    """Compare average daily sales on promotion vs non-promotion days."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = df["promotion"].map({1: "Promotion", 0: "No promotion"})
    sns.barplot(x=labels, y=df[config.TARGET_COL], ax=axes[0], hue=labels,
                palette="Set2", legend=False, errorbar="sd")
    axes[0].set_title("Average Sales: Promotion vs No Promotion")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average units sold per record")

    promo_by_cat = (
        df.groupby(["product_category", "promotion"])[config.TARGET_COL]
        .mean()
        .reset_index()
    )
    promo_by_cat["promotion"] = promo_by_cat["promotion"].map({1: "Promotion", 0: "No promotion"})
    sns.barplot(data=promo_by_cat, x="product_category", y=config.TARGET_COL,
                hue="promotion", ax=axes[1], palette="Set2")
    axes[1].set_title("Promotion Effect by Category")
    axes[1].set_xlabel("Product category")
    axes[1].set_ylabel("Average units sold")
    axes[1].legend(title="")

    _save(fig, config.CHART_PROMOTION)


# ----------------------------------------------------------------------
# 8. Stock availability analysis
# ----------------------------------------------------------------------
def plot_stock_analysis(df: pd.DataFrame) -> None:
    """How stock relates to sales, and how often stock ran low."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(df["stock_available"], df[config.TARGET_COL],
                    alpha=0.15, s=8, color="#2a6f97")
    axes[0].set_title("Stock Available vs Sales Quantity")
    axes[0].set_xlabel("Stock available (units)")
    axes[0].set_ylabel("Sales quantity (units)")

    # Days where stock was lower than 20% of that product's average stock
    stock_ratio = df["stock_available"] / df.groupby(config.GROUP_COLS)["stock_available"].transform("mean")
    low_stock = (stock_ratio < 0.2).map({True: "Low stock (<20% of normal)", False: "Normal stock"})
    counts = low_stock.value_counts()
    axes[1].bar(counts.index, counts.values, color=["#e07a5f", "#81b29a"])
    axes[1].set_title("Low Stock Frequency")
    axes[1].set_ylabel("Number of records")
    for i, value in enumerate(counts.values):
        axes[1].text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    _save(fig, config.CHARTS_DIR / "05b_stock_analysis.png")


def run_full_eda(df: pd.DataFrame) -> None:
    """Run every EDA step in order. Used by main.py and by the notebook."""
    dataset_overview(df)
    print("\n--- CREATING EDA CHARTS ---")
    plot_sales_distribution(df)
    plot_daily_trend(df)
    plot_monthly_trend(df)
    plot_category_sales(df)
    plot_store_sales(df)
    plot_promotion_effect(df)
    plot_stock_analysis(df)
