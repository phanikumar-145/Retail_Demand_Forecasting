"""
forecasting.py
--------------
STEP 5 of the pipeline: forecast demand for the next 7 days.

THE APPROACH (recursive / multi-step forecasting)
    Our model needs lag features (sales 1, 7 and 30 days ago). For tomorrow we
    know those values from real history. But for day 2 we do NOT know day 1's
    real sales yet - so we feed the model its OWN day-1 prediction and use that
    as lag_1, then predict day 2, and so on. This is called recursive forecasting.

    For every future date we therefore build the features from:
        - the real sales history BEFORE the forecast origin, plus
        - the predictions we already made for earlier forecast days.

LIMITATIONS (documented honestly, also repeated in the README)
    1. Errors accumulate: a bad prediction for day 1 becomes a wrong input for
       day 2. The further ahead we forecast, the less reliable the number is.
       A 1-day forecast is much more trustworthy than a 7-day forecast.
    2. We assume NO promotions in the future (we cannot know the promo calendar).
    3. We hold the unit price constant at the last known price.
    4. We ignore stock availability when predicting, because stock is not known
       in advance; stock is only used later in the replenishment step.
    5. The model cannot predict sudden shocks (festivals, strikes, new
       competitors). It repeats the patterns it saw in the two years of history.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


def _features_for_one_day(history: list, target_date: pd.Timestamp,
                          promotion: int, unit_price: float) -> dict:
    """
    Build one feature row for a single future date.

    'history' must be the chronological list of sales values up to the day
    BEFORE target_date (real + already-predicted values).
    This mirrors exactly what .shift(1) did during training:
        lag_1          = sales on day T-1
        rolling_mean_7 = mean of sales on days T-7 ... T-1
    """
    if len(history) < 30:
        raise ValueError("Need at least 30 days of history to build lag_30 / rolling_mean_30.")

    return {
        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
        "day_of_week": target_date.dayofweek,
        "is_weekend": int(target_date.dayofweek in (5, 6)),
        "promotion": promotion,
        "unit_price": unit_price,
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_30": history[-30],
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_30": float(np.mean(history[-30:])),
    }


def forecast_future_demand(cleaned_df: pd.DataFrame, model,
                           horizon: int = config.FORECAST_HORIZON,
                           feature_columns: list | None = None) -> pd.DataFrame:
    """
    Forecast demand for the next `horizon` days for every store-product pair.

    Returns a DataFrame with:
        date, store_id, product_id, product_category, predicted_demand
    """
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    df = cleaned_df.sort_values(config.GROUP_COLS + [config.DATE_COL])
    last_date = df[config.DATE_COL].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    print(f"\nForecast origin (last real data): {last_date.date()}")
    print(f"Forecasting {horizon} days: {future_dates[0].date()} -> {future_dates[-1].date()}")

    rows = []
    series_count = 0

    for (store_id, product_id), group in df.groupby(config.GROUP_COLS):
        group = group.sort_values(config.DATE_COL)
        category = group["product_category"].iloc[-1]

        # Real history up to the forecast origin (float list, oldest -> newest)
        history = group[config.TARGET_COL].astype(float).tolist()
        # Price is assumed to stay at the last known value
        last_price = float(group["unit_price"].iloc[-1])

        for future_date in future_dates:
            features = _features_for_one_day(
                history=history,
                target_date=future_date,
                promotion=0,          # assumption: no promotions scheduled
                unit_price=last_price,
            )
            # Keep the exact training column order in a one-row DataFrame
            X_future = pd.DataFrame([features])[feature_columns]

            prediction = float(model.predict(X_future)[0])
            prediction = max(prediction, 0.0)          # demand cannot be negative
            prediction = int(round(prediction))

            rows.append(
                {
                    "date": future_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "product_category": category,
                    "predicted_demand": prediction,
                }
            )

            # Feed the prediction back in so the NEXT day can use it as lag_1
            history.append(prediction)

        series_count += 1

    forecasts = pd.DataFrame(rows)
    print(f"Forecasts created for {series_count} store-product series "
          f"({len(forecasts)} rows).")
    print(f"Average predicted demand per row: {forecasts['predicted_demand'].mean():.2f} units")

    return forecasts


def save_forecasts(forecasts: pd.DataFrame) -> None:
    """Save outputs/forecasts.csv"""
    config.ensure_directories()
    forecasts.to_csv(config.FORECAST_FILE, index=False, encoding="utf-8")
    print(f"Forecasts saved to: {config.FORECAST_FILE}")


def plot_forecasts(forecasts: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """
    Chart #8: forecasted demand.

    Left  - total forecast per day (all stores) - what the business needs to plan.
    Right - one example store-product: last 30 real days + 7 forecast days.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Left: total predicted demand per future day ----
    daily_forecast = forecasts.groupby("date")["predicted_demand"].sum().reset_index()
    axes[0].bar(daily_forecast["date"].dt.strftime("%a %d %b"),
                daily_forecast["predicted_demand"], color="#2a6f97")
    axes[0].set_title("Total Forecasted Demand - Next 7 Days (All Stores)")
    axes[0].set_xlabel("Forecast date")
    axes[0].set_ylabel("Predicted units")
    for i, value in enumerate(daily_forecast["predicted_demand"]):
        axes[0].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    # ---- Right: one example series ----
    example = (
        forecasts.groupby(["store_id", "product_id"])["predicted_demand"].sum().idxmax()
    )
    example_forecast = forecasts[
        (forecasts["store_id"] == example[0]) & (forecasts["product_id"] == example[1])
    ].sort_values("date")

    history = cleaned_df[
        (cleaned_df["store_id"] == example[0]) & (cleaned_df["product_id"] == example[1])
    ].sort_values(config.DATE_COL).tail(30)

    axes[1].plot(history[config.DATE_COL], history[config.TARGET_COL],
                 marker="o", markersize=3, color="#1d3557", label="Actual sales (last 30 days)")
    axes[1].plot(example_forecast["date"], example_forecast["predicted_demand"],
                 marker="o", markersize=5, color="#e63946", linestyle="--",
                 label="Forecast (next 7 days)")
    axes[1].axvline(history[config.DATE_COL].max(), color="grey", linestyle=":", linewidth=1.5)
    axes[1].set_title(f"Forecast Example - Store {example[0]}, Product {example[1]}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Units")
    axes[1].legend()
    for label in axes[1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    fig.suptitle("Forecasted Demand", fontsize=15)
    config.ensure_directories()
    fig.tight_layout()
    fig.savefig(config.CHART_FORECAST, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart saved -> {config.CHART_FORECAST.name}")
