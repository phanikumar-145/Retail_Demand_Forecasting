"""
data_preprocessing.py
---------------------
STEP 1 of the pipeline: get the data and clean it.

Two jobs live here:
    1. create_synthetic_dataset() -> builds a realistic 2-year retail dataset
       (only used when the user has not supplied data/raw/retail_sales.csv).
    2. clean_data() -> fixes dates, duplicates, missing values and invalid values.

Run this file directly to just create + clean the data:
    python -m src.data_preprocessing
"""

import numpy as np
import pandas as pd

from . import config


# ======================================================================
# PART A: CREATE A SYNTHETIC DATASET
# ======================================================================
def create_synthetic_dataset(
    start_date: str = config.SYNTHETIC_START_DATE,
    end_date: str = config.SYNTHETIC_END_DATE,
    n_stores: int = config.SYNTHETIC_N_STORES,
    n_products: int = config.SYNTHETIC_N_PRODUCTS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Build a realistic-looking daily retail dataset.

    The data is SYNTHETIC (invented by code), not real business data.
    It still contains believable patterns so a model can learn from it:
        - weekly pattern   : weekends sell more
        - yearly pattern   : a festive peak around Oct-Dec
        - slow growth      : a small upward trend
        - promotions       : random promotions lift sales
        - stock limits     : low stock caps how much can be sold
        - noise            : random day-to-day variation
    It also deliberately contains a few "dirty" rows (missing values,
    duplicates, negative sales, bad dates) so preprocessing has real work to do.
    """
    rng = np.random.default_rng(random_state)  # fixed seed -> reproducible data

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # ---- Stores: each store has a different overall size factor ----
    stores = [f"S{i:02d}" for i in range(1, n_stores + 1)]
    store_factor = dict(zip(stores, [1.00, 1.25, 0.80, 1.10][:n_stores]))

    # ---- Categories and products ----
    categories = ["Beverages", "Dairy", "Household", "Snacks"]
    products = []  # list of small dicts describing each product
    for i in range(1, n_products + 1):
        category = categories[(i - 1) % len(categories)]
        base_demand = {
            "Beverages": 55,
            "Dairy": 70,
            "Household": 30,
            "Snacks": 45,
        }[category]
        price = {
            "Beverages": 45.0,
            "Dairy": 60.0,
            "Household": 130.0,
            "Snacks": 35.0,
        }[category]
        products.append(
            {
                "product_id": f"P{i:03d}",
                # random variation between products so they are not identical
                "product_category": category,
                "base_demand": base_demand * rng.uniform(0.7, 1.35),
                "unit_price": round(price * rng.uniform(0.85, 1.2), 2),
                "promo_lift": rng.uniform(1.25, 1.70),   # how strongly promos help
            }
        )

    rows = []
    for store in stores:
        for p in products:
            # ---- Base level for this store-product pair ----
            base = p["base_demand"] * store_factor[store]

            for d in dates:
                # (a) trend: demand grows slowly over the 2 years
                days_elapsed = (d - dates[0]).days
                trend = 1.0 + 0.00025 * days_elapsed

                # (b) weekly seasonality: weekend uplift
                weekly = 1.18 if d.dayofweek in (5, 6) else 1.0

                # (c) yearly seasonality: festive peak in Oct-Dec, dip in Feb
                yearly = 1.0 + 0.30 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365)
                if d.month in (10, 11, 12):
                    yearly *= 1.15

                # (d) promotions: more likely on weekends
                promo_prob = 0.20 if d.dayofweek in (5, 6) else 0.10
                promotion = int(rng.random() < promo_prob)
                promo_mult = p["promo_lift"] if promotion == 1 else 1.0

                # (e) expected demand + random noise
                expected = base * trend * weekly * yearly * promo_mult
                noise = rng.normal(1.0, 0.18)          # +/- 18% random noise
                quantity = expected * max(noise, 0.05)

                # (f) stock: a few days of cover, so stock-outs happen sometimes
                stock = int(max(0, quantity * rng.uniform(1.5, 6.0)))

                # (g) a stock-out limits the sales we can actually record
                sold = int(round(min(quantity, stock)))

                # (h) price moves very slightly over time
                price = round(p["unit_price"] * (1 + 0.0001 * days_elapsed), 2)

                rows.append(
                    {
                        "date": d,
                        "store_id": store,
                        "product_id": p["product_id"],
                        "product_category": p["product_category"],
                        "sales_quantity": sold,
                        "unit_price": price,
                        "promotion": promotion,
                        "stock_available": stock,
                    }
                )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Deliberately add some bad rows, so cleaning is not pointless.
    # ------------------------------------------------------------------
    n = len(df)

    # 1. ~1% missing values in some columns
    for col, frac in [("sales_quantity", 0.010), ("unit_price", 0.008),
                      ("stock_available", 0.010), ("promotion", 0.005)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2. ~0.4% exact duplicate rows
    dup_idx = rng.choice(n, size=int(n * 0.004), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # 3. ~0.3% negative sales (impossible -> data entry errors)
    neg_idx = rng.choice(df.index, size=int(n * 0.003), replace=False)
    df.loc[neg_idx, "sales_quantity"] = -df.loc[neg_idx, "sales_quantity"].fillna(5)

    # 4. a few impossible dates written in the wrong format
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    bad_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[bad_idx, "date"] = "31/02/2024"   # 31 February does not exist

    # 5. shuffle the row order (real exports are rarely perfectly sorted)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df


# ======================================================================
# PART B: LOAD
# ======================================================================
def load_raw_data(regenerate: bool = False) -> pd.DataFrame:
    """
    Load data/raw/retail_sales.csv.

    If the file is missing (or regenerate=True), a synthetic dataset is created
    and saved first. Raises a clear error if the file exists but is unusable.
    """
    config.ensure_directories()

    if regenerate or not config.RAW_DATA_FILE.exists():
        print("No raw dataset found -> creating a SYNTHETIC retail dataset...")
        raw_df = create_synthetic_dataset()
        raw_df.to_csv(config.RAW_DATA_FILE, index=False, encoding="utf-8")
        print(f"Synthetic dataset saved to: {config.RAW_DATA_FILE}  ({len(raw_df):,} rows)")

    try:
        df = pd.read_csv(config.RAW_DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the dataset at {config.RAW_DATA_FILE}. "
            "Place your retail_sales.csv there, or delete it so it is regenerated."
        )
    except Exception as error:
        raise ValueError(f"Could not read {config.RAW_DATA_FILE}: {error}")

    if df.empty:
        raise ValueError(f"The dataset at {config.RAW_DATA_FILE} is empty.")

    # Normalise column names: lowercase + strip spaces, so 'Date ' becomes 'date'
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {config.REQUIRED_COLUMNS}"
        )

    return df


# ======================================================================
# PART C: CLEAN
# ======================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data step by step and return a tidy DataFrame.

    Cleaning steps (each one prints what it did, so nothing is hidden):
        1. Fix data types (date -> datetime, numbers -> numeric)
        2. Drop rows with impossible dates
        3. Remove exact duplicate rows
        4. Handle negative / invalid values
        5. Fill missing values sensibly
        6. Clip outliers that are clearly data-entry mistakes
    """
    df = df.copy()
    print("\n--- CLEANING DATA ---")
    print(f"Rows in raw data: {len(df):,}")

    # ---- Step 1: data types -------------------------------------------
    # errors="coerce" turns unparseable text (e.g. "31/02/2024") into NaT,
    # which we then drop, instead of crashing the whole pipeline.
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")

    for col in ["sales_quantity", "unit_price", "promotion", "stock_available"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Step 2: impossible dates --------------------------------------
    bad_dates = df[config.DATE_COL].isna().sum()
    if bad_dates:
        print(f"Dropped {bad_dates} rows with invalid/missing dates.")
        df = df.dropna(subset=[config.DATE_COL])

    # ---- Step 3: duplicate records -------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} exact duplicate rows.")

    # Duplicate (date, store, product) keys would confuse the model: keep the first.
    before = len(df)
    df = df.drop_duplicates(subset=[config.DATE_COL, "store_id", "product_id"], keep="first")
    if before != len(df):
        print(f"Removed {before - len(df)} duplicate (date, store, product) rows.")

    # Store/product text columns: strip spaces, upper-case the ids
    df["store_id"] = df["store_id"].astype(str).str.strip().str.upper()
    df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()
    df["product_category"] = df["product_category"].astype(str).str.strip()

    # ---- Step 4: invalid / negative values -----------------------------
    # Negative sales are impossible -> treat as missing, then impute below.
    negative_sales = (df[config.TARGET_COL] < 0).sum()
    df.loc[df[config.TARGET_COL] < 0, config.TARGET_COL] = np.nan
    print(f"Found {negative_sales} negative sales values (set to missing).")

    # Negative stock is impossible -> treat as missing.
    negative_stock = (df["stock_available"] < 0).sum()
    df.loc[df["stock_available"] < 0, "stock_available"] = np.nan
    print(f"Found {negative_stock} negative stock values (set to missing).")

    # A price of zero or below is invalid -> treat as missing.
    df.loc[df["unit_price"] <= 0, "unit_price"] = np.nan

    # promotion must be 0 or 1
    df["promotion"] = df["promotion"].round()
    df.loc[~df["promotion"].isin([0, 1]), "promotion"] = np.nan

    # ---- Step 5: missing values ----------------------------------------
    print("\nMissing values before filling:")
    print(df[config.REQUIRED_COLUMNS].isna().sum().to_string())

    # sales_quantity -> median of the same store+product (that product's normal level)
    df[config.TARGET_COL] = df.groupby(config.GROUP_COLS)[config.TARGET_COL].transform(
        lambda s: s.fillna(s.median())
    )
    # if a whole store-product is missing, fall back to the product's median
    df[config.TARGET_COL] = df.groupby("product_id")[config.TARGET_COL].transform(
        lambda s: s.fillna(s.median())
    )
    df[config.TARGET_COL] = df[config.TARGET_COL].fillna(df[config.TARGET_COL].median())

    # unit_price -> median price of the same product
    df["unit_price"] = df.groupby("product_id")["unit_price"].transform(
        lambda s: s.fillna(s.median())
    )
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    # stock_available -> median of the same store+product, else 0
    df["stock_available"] = df.groupby(config.GROUP_COLS)["stock_available"].transform(
        lambda s: s.fillna(s.median())
    )
    df["stock_available"] = df["stock_available"].fillna(0)

    # promotion -> assume "no promotion" when unknown
    df["promotion"] = df["promotion"].fillna(0)

    print("\nMissing values after filling:")
    print(df[config.REQUIRED_COLUMNS].isna().sum().to_string())

    # ---- Step 6: final tidy-up ------------------------------------------
    df[config.TARGET_COL] = df[config.TARGET_COL].round().clip(lower=0).astype(int)
    df["stock_available"] = df["stock_available"].round().clip(lower=0).astype(int)
    df["promotion"] = df["promotion"].astype(int)
    df["unit_price"] = df["unit_price"].round(2)

    # Extreme sales values (e.g. 50x the product median) are data-entry errors:
    # cap them so a handful of rows cannot dominate the model.
    median = df.groupby(config.GROUP_COLS)[config.TARGET_COL].transform("median")
    upper_limit = (median * 10).clip(lower=1)
    outliers = (df[config.TARGET_COL] > upper_limit).sum()
    df[config.TARGET_COL] = np.minimum(df[config.TARGET_COL], upper_limit).astype(int)
    if outliers:
        print(f"Capped {outliers} extreme sales values at 10x the product median.")

    # Sort chronologically: needed for lag/rolling features later.
    df = df.sort_values([config.DATE_COL, "store_id", "product_id"]).reset_index(drop=True)

    print("\nData types after cleaning:")
    print(df.dtypes.to_string())
    print(f"\nFinal cleaned rows: {len(df):,}")
    print(f"Date range: {df[config.DATE_COL].min().date()} -> {df[config.DATE_COL].max().date()}")
    print(f"Stores: {df['store_id'].nunique()} | Products: {df['product_id'].nunique()} "
          f"| Categories: {df['product_category'].nunique()}")

    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset to data/processed/retail_sales_cleaned.csv."""
    config.ensure_directories()
    df.to_csv(config.CLEANED_DATA_FILE, index=False, encoding="utf-8")
    print(f"\nCleaned dataset saved to: {config.CLEANED_DATA_FILE}")


def run_preprocessing(regenerate: bool = False) -> pd.DataFrame:
    """Convenience wrapper: load -> clean -> save. Returns the cleaned DataFrame."""
    raw_df = load_raw_data(regenerate=regenerate)
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df)
    return cleaned_df


if __name__ == "__main__":
    run_preprocessing()
