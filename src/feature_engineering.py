"""
feature_engineering.py
----------------------
STEP 2 of the pipeline: turn the cleaned table into model-ready features.

WHAT WE CREATE
    Calendar features : year, month, day, day_of_week, is_weekend
    Known drivers     : promotion, unit_price
    Lag features      : lag_1, lag_7, lag_30          (past sales)
    Rolling averages  : rolling_mean_7, rolling_mean_30

WHY THESE FEATURES
    Calendar features let the model learn weekly and yearly habits
    (weekends are busier, the festive season is busier).
    Lag features are the strongest signal in retail: what sold yesterday,
    last week and last month is a very good guess for what sells today.
    Rolling averages smooth out noise and show the recent level of demand.

HOW WE AVOID DATA LEAKAGE (the most important idea in this file)
    1. Rows are sorted by store, product and date first.
    2. Every lag/rolling feature is computed with .shift(1), so the value
       for day T only uses information from day T-1 and earlier.
       Without .shift(1), rolling_mean_7 for day T would contain day T's own
       sales -> the model would "see the answer" and look perfect in training
       but fail on real future data.
    3. The train/test split is CHRONOLOGICAL: the model trains on the first
       80% of dates and is tested on the last 20%. A random_split would put
       future rows in the training set and past rows in the test set.
"""

import numpy as np
import pandas as pd

from . import config


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the date column."""
    df = df.copy()
    # Make sure the date column really is a datetime (defensive check)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    df["year"] = df[config.DATE_COL].dt.year
    df["month"] = df[config.DATE_COL].dt.month
    df["day"] = df[config.DATE_COL].dt.day
    # dayofweek: Monday=0 ... Sunday=6
    df["day_of_week"] = df[config.DATE_COL].dt.dayofweek
    # weekend = Saturday or Sunday -> 1, otherwise 0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-average features, computed separately per store+product.

    groupby(GROUP_COLS) is what makes this per product per store:
    the sales of product P001 in store S01 never mix with S02's numbers.
    """
    df = df.copy()

    # Sort chronologically inside each store-product series. Without sorting,
    # .shift() would point at the wrong row.
    df = df.sort_values(config.GROUP_COLS + [config.DATE_COL]).reset_index(drop=True)

    grouped = df.groupby(config.GROUP_COLS)[config.TARGET_COL]

    # ---- Lag features: sales 1 / 7 / 30 days ago -----------------------
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_30"] = grouped.shift(30)

    # ---- Rolling averages of PAST sales only ---------------------------
    # .shift(1) FIRST, then .rolling(): the window covers days T-7..T-1,
    # never day T itself. This is the anti-leakage step.
    shifted_sales = grouped.shift(1)
    df["rolling_mean_7"] = (
        shifted_sales.groupby([df["store_id"], df["product_id"]])
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_mean_30"] = (
        shifted_sales.groupby([df["store_id"], df["product_id"]])
        .rolling(window=30, min_periods=30)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Rounding keeps the CSV files readable
    for col in ["rolling_mean_7", "rolling_mean_30"]:
        df[col] = df[col].round(3)

    # The first 30 days of every store-product series have no lag_30 /
    # rolling_mean_30 yet (NaN). Those rows are dropped, not filled,
    # because inventing past sales would be fake information.
    before = len(df)
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET_COL]).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} early rows that had no lag history yet.")

    return df


def build_features(cleaned_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Full feature pipeline: calendar features + lag/rolling features."""
    print("\n--- FEATURE ENGINEERING ---")
    df = create_time_features(cleaned_df)
    df = create_lag_and_rolling_features(df)
    df["unit_price"] = df["unit_price"].astype(float)
    print(f"Feature table ready: {len(df):,} rows x {len(config.FEATURE_COLUMNS)} features")
    print("Features used by the models:")
    print("  " + ", ".join(config.FEATURE_COLUMNS))

    if save:
        config.ensure_directories()
        feature_file = config.PROCESSED_DATA_DIR / "retail_sales_features.csv"
        df.to_csv(feature_file, index=False, encoding="utf-8")
        print(f"Feature table saved to: {feature_file}")

    return df


def chronological_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """
    Split by DATE, not randomly.

    The first (1 - test_fraction) of the dates become the training set and
    the last test_fraction become the test set. This copies the real world:
    we train on the past and predict the future.

    Returns: X_train, X_test, y_train, y_test, train_df, test_df
    """
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)

    unique_dates = np.sort(df[config.DATE_COL].unique())
    cut_index = int(len(unique_dates) * (1 - test_fraction))
    cut_date = unique_dates[cut_index]

    train_df = df[df[config.DATE_COL] < cut_date].copy()
    test_df = df[df[config.DATE_COL] >= cut_date].copy()

    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COL]

    print("\n--- CHRONOLOGICAL TRAIN/TEST SPLIT ---")
    print(f"Train: {train_df[config.DATE_COL].min().date()} -> "
          f"{train_df[config.DATE_COL].max().date()}  ({len(train_df):,} rows)")
    print(f"Test : {test_df[config.DATE_COL].min().date()} -> "
          f"{test_df[config.DATE_COL].max().date()}  ({len(test_df):,} rows)")

    return X_train, X_test, y_train, y_test, train_df, test_df
