"""
config.py
---------
One central place for every file path and setting used in the project.

Why this file exists:
    If we hardcode paths like "outputs/forecasts.csv" in five different files,
    the project breaks the moment we move a folder. Here we build the paths
    ONE time from the project root, and every other module imports them.
    The paths work on Windows, macOS and Linux because we use pathlib.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 1. PROJECT ROOT
# ----------------------------------------------------------------------
# __file__            -> .../retail-demand-forecasting/src/config.py
# .resolve().parent   -> .../retail-demand-forecasting/src
# .parent             -> .../retail-demand-forecasting   (the project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------
# 2. FOLDERS  (created automatically if they do not exist)
# ----------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

ALL_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    CHARTS_DIR,
    NOTEBOOKS_DIR,
]

# ----------------------------------------------------------------------
# 3. FILES
# ----------------------------------------------------------------------
# ---- Input / intermediate data ----
RAW_DATA_FILE = RAW_DATA_DIR / "retail_sales.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "retail_sales_cleaned.csv"

# ---- Saved model artifacts ----
MODEL_FILE = MODELS_DIR / "best_model.joblib"          # the trained model itself
FEATURES_FILE = MODELS_DIR / "model_features.joblib"   # exact column order used in training
METADATA_FILE = MODELS_DIR / "model_metadata.joblib"   # model name, metrics, split dates

# ---- Output tables ----
FORECAST_FILE = OUTPUTS_DIR / "forecasts.csv"
EVALUATION_FILE = OUTPUTS_DIR / "evaluation_report.csv"
CATEGORY_ERROR_FILE = OUTPUTS_DIR / "category_errors.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
REORDER_FILE = OUTPUTS_DIR / "reorder_recommendations.csv"

# ---- Output charts ----
CHART_DAILY_TREND = CHARTS_DIR / "01_daily_sales_trend.png"
CHART_MONTHLY_TREND = CHARTS_DIR / "02_monthly_sales_trend.png"
CHART_CATEGORY_SALES = CHARTS_DIR / "03_category_wise_sales.png"
CHART_STORE_SALES = CHARTS_DIR / "04_store_wise_sales.png"
CHART_PROMOTION = CHARTS_DIR / "05_promotion_vs_non_promotion.png"
CHART_ACTUAL_VS_PREDICTED = CHARTS_DIR / "06_actual_vs_predicted.png"
CHART_MODEL_COMPARISON = CHARTS_DIR / "07_model_comparison.png"
CHART_FORECAST = CHARTS_DIR / "08_forecasted_demand.png"
CHART_CATEGORY_ERRORS = CHARTS_DIR / "09_category_wise_errors.png"
CHART_STOCK_VS_REORDER = CHARTS_DIR / "10_stock_vs_reorder.png"

# ----------------------------------------------------------------------
# 4. COLUMN AND BUSINESS SETTINGS
# ----------------------------------------------------------------------
DATE_COL = "date"
TARGET_COL = "sales_quantity"
GROUP_COLS = ["store_id", "product_id"]

REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "product_category",
    "sales_quantity",
    "unit_price",
    "promotion",
    "stock_available",
]

# The exact features the models are trained on (order matters at prediction time).
FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "promotion",
    "unit_price",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]

# ----------------------------------------------------------------------
# 5. REPRODUCIBILITY AND MODEL SETTINGS
# ----------------------------------------------------------------------
RANDOM_STATE = 42          # fixed seed -> same results every run
TEST_FRACTION = 0.20       # last 20% of DATES are used for testing (chronological split)
FORECAST_HORIZON = 7       # forecast the next 7 days

# ----------------------------------------------------------------------
# 6. INVENTORY / REPLENISHMENT ASSUMPTIONS
# ----------------------------------------------------------------------
DEFAULT_LEAD_TIME_DAYS = 3   # assumed supplier lead time (documented assumption)
SAFETY_STOCK_Z = 1.65        # z-score for ~95% service level
DEMAND_STD_WINDOW = 30       # use last 30 days of real sales to measure demand variability

# ----------------------------------------------------------------------
# 7. SYNTHETIC DATA SETTINGS (only used when no real dataset is supplied)
# ----------------------------------------------------------------------
SYNTHETIC_START_DATE = "2023-01-01"
SYNTHETIC_END_DATE = "2024-12-31"
SYNTHETIC_N_STORES = 4
SYNTHETIC_N_PRODUCTS = 12


def ensure_directories() -> None:
    """Create every project folder that is missing. Safe to call many times."""
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
