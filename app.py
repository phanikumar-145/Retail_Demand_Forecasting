"""
app.py
======
Streamlit dashboard for the Retail Demand Forecasting and Replenishment System.

IMPORTANT: this file contains NO machine learning. It never trains a model.
It only READS the files produced by `python main.py`:

    data/processed/retail_sales_cleaned.csv   -> historical sales
    outputs/forecasts.csv                     -> 7-day demand forecast
    outputs/evaluation_report.csv             -> model comparison (MAE/RMSE/R2)
    outputs/category_errors.csv               -> errors per product category
    outputs/test_predictions.csv              -> actual vs predicted on the test set
    outputs/reorder_recommendations.csv       -> reorder quantities
    models/best_model.joblib                  -> the trained model (shown, not retrained)

Run with:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Paths (same convention as src/config.py, without importing the ML code)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"

CLEANED_FILE = PROCESSED_DIR / "retail_sales_cleaned.csv"
FORECAST_FILE = OUTPUTS_DIR / "forecasts.csv"
EVALUATION_FILE = OUTPUTS_DIR / "evaluation_report.csv"
CATEGORY_ERROR_FILE = OUTPUTS_DIR / "category_errors.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
REORDER_FILE = OUTPUTS_DIR / "reorder_recommendations.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.joblib"

PRIMARY = "#2a6f97"
ACCENT = "#e63946"

st.set_page_config(
    page_title="Retail Demand Forecasting & Replenishment",
    page_icon="📦",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV from disk. Cached so the file is not re-read on every click."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clean_sales() -> pd.DataFrame:
    """Historical cleaned sales, with the date column parsed."""
    df = pd.read_csv(CLEANED_FILE, parse_dates=["date"])
    return df


@st.cache_resource(show_spinner=False)
def load_model_metadata() -> dict:
    """Model metadata saved by the training step (name, metrics, split dates)."""
    if not METADATA_FILE.exists():
        return {}
    try:
        return joblib.load(METADATA_FILE)
    except Exception:
        return {}


def required_files_exist() -> bool:
    """True only when every output the dashboard needs is present."""
    needed = [
        CLEANED_FILE, FORECAST_FILE, EVALUATION_FILE,
        CATEGORY_ERROR_FILE, REORDER_FILE, TEST_PREDICTIONS_FILE,
    ]
    return all(path.exists() for path in needed)


def run_pipeline() -> None:
    """
    Run `python main.py` as a separate process.

    Doing it in a subprocess (instead of importing the functions) keeps the
    dashboard completely separate from the ML logic and avoids blocking the
    Streamlit session with a 3-minute training job inside the UI process.
    """
    with st.spinner("Running the ML pipeline (python main.py). This takes about 3 minutes..."):
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.success("Pipeline finished successfully. Reloading outputs...")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("The pipeline failed. Details below.")
        st.code((result.stderr or result.stdout)[-4000:])


# ----------------------------------------------------------------------
# Reusable UI pieces
# ----------------------------------------------------------------------
def kpi_row(items: list) -> None:
    """Draw a row of KPI cards: items = [(label, value, help_text), ...]"""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label=label, value=value, help=help_text)


def download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    """CSV download button."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


# ----------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------------------------
def page_overview() -> None:
    page_header(
        "📦 Overview",
        "Numbers read directly from the pipeline outputs — nothing is hardcoded.",
    )

    sales = load_clean_sales()
    reorder = load_csv(str(REORDER_FILE))
    metadata = load_model_metadata()

    total_sales = int(sales["sales_quantity"].sum())
    n_products = sales["product_id"].nunique()
    n_stores = sales["store_id"].nunique()
    total_stock = int(reorder["current_stock"].sum())
    n_restock = int((reorder["recommended_reorder_quantity"] > 0).sum())
    total_units_to_order = int(reorder["recommended_reorder_quantity"].sum())

    kpi_row(
        [
            ("Total Sales (units)", f"{total_sales:,}", "Sum of sales_quantity over the full 2-year history"),
            ("Products", f"{n_products}", "Unique product IDs in the dataset"),
            ("Stores", f"{n_stores}", "Unique store IDs in the dataset"),
            ("Current Stock (units)", f"{total_stock:,}", "Latest recorded stock per store-product"),
            ("Need Restock", f"{n_restock}", "Store-products with a reorder quantity above 0"),
        ]
    )

    st.subheader("What this system does")
    st.markdown(
        """
        1. **Clean** two years of daily retail sales (duplicates, missing values, invalid rows).
        2. **Explore** the data with charts — trends, categories, stores, promotions, stock.
        3. **Engineer** time features (lags and rolling averages) without data leakage.
        4. **Train and compare** three models and pick the best one on the test period.
        5. **Forecast** demand for the next 7 days for every store and product.
        6. **Recommend** reorder quantities using forecast, stock, lead time and safety stock.
        """
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Sales over time (all stores)")
        daily = sales.groupby("date")["sales_quantity"].sum().reset_index()
        fig = px.line(daily, x="date", y="sales_quantity",
                      labels={"date": "Date", "sales_quantity": "Units sold"})
        fig.update_traces(line_color=PRIMARY)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("System snapshot")
        if metadata:
            st.write(f"**Selected model:** {metadata.get('model_name', 'n/a')}")
            metrics = metadata.get("test_metrics", {})
            st.write(f"**Test MAE:** {metrics.get('MAE', 0):.2f} units")
            st.write(f"**Test RMSE:** {metrics.get('RMSE', 0):.2f} units")
            st.write(f"**Test R²:** {metrics.get('R2', 0):.4f}")
            st.write(f"**Trained on:** {metadata.get('train_start')} → {metadata.get('train_end')}")
            st.write(f"**Tested on:** {metadata.get('test_start')} → {metadata.get('test_end')}")
        else:
            st.info("Model metadata not found. Run the pipeline to create it.")

        st.write(f"**Forecast horizon:** 7 days")
        st.write(f"**Units recommended to order:** {total_units_to_order:,}")
        st.write(f"**Lead time assumption:** "
                 f"{int(reorder['lead_time_days'].iloc[0])} days")

    st.subheader("Top 10 reorder priorities")
    top = reorder.head(10)[
        ["store_id", "product_id", "product_category", "current_stock",
         "expected_demand", "safety_stock", "recommended_reorder_quantity"]
    ]
    st.dataframe(top, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# PAGE 2: SALES ANALYSIS
# ----------------------------------------------------------------------
def page_sales_analysis() -> None:
    page_header("📈 Sales Analysis", "Historical sales from data/processed/retail_sales_cleaned.csv")

    sales = load_clean_sales()

    # ---- Filters ----
    left, mid, right = st.columns(3)
    stores = sorted(sales["store_id"].unique())
    categories = sorted(sales["product_category"].unique())
    selected_stores = left.multiselect("Store", stores, default=stores)
    selected_categories = mid.multiselect("Category", categories, default=categories)
    date_range = right.date_input(
        "Date range",
        value=(sales["date"].min().date(), sales["date"].max().date()),
        min_value=sales["date"].min().date(),
        max_value=sales["date"].max().date(),
    )

    if not selected_stores or not selected_categories:
        st.warning("Select at least one store and one category.")
        return

    filtered = sales[
        sales["store_id"].isin(selected_stores)
        & sales["product_category"].isin(selected_categories)
    ]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    if filtered.empty:
        st.warning("No data for this selection.")
        return

    kpi_row(
        [
            ("Total Sales (units)", f"{int(filtered['sales_quantity'].sum()):,}", "Units sold in the selection"),
            ("Average Daily Sales", f"{filtered.groupby('date')['sales_quantity'].sum().mean():,.1f}", "Average units per day"),
            ("Promotion Days", f"{int(filtered['promotion'].sum()):,}", "Records flagged as promotion"),
            ("Average Stock", f"{filtered['stock_available'].mean():,.1f}", "Average units available"),
        ]
    )

    # ---- Daily trend ----
    st.subheader("Daily sales trend")
    daily = filtered.groupby("date")["sales_quantity"].sum().reset_index()
    daily["7-day average"] = daily["sales_quantity"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["sales_quantity"],
                             name="Daily sales", line=dict(color="#8ecae6", width=1)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["7-day average"],
                             name="7-day average", line=dict(color=ACCENT, width=2.5)))
    fig.update_layout(height=360, xaxis_title="Date", yaxis_title="Units sold",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Monthly trend ----
    left, right = st.columns(2)
    with left:
        st.subheader("Monthly sales trend")
        monthly = (
            filtered.assign(month=filtered["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")["sales_quantity"].sum().reset_index()
        )
        fig = px.bar(monthly, x="month", y="sales_quantity",
                     labels={"month": "Month", "sales_quantity": "Units sold"})
        fig.update_traces(marker_color="#457b9d")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Category-wise sales")
        by_category = (
            filtered.groupby("product_category")["sales_quantity"].sum()
            .sort_values(ascending=True).reset_index()
        )
        fig = px.bar(by_category, x="sales_quantity", y="product_category",
                     orientation="h", color="sales_quantity", color_continuous_scale="Viridis",
                     labels={"sales_quantity": "Units sold", "product_category": ""})
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Store-wise + promotion ----
    left, right = st.columns(2)
    with left:
        st.subheader("Store-wise sales")
        by_store = filtered.groupby("store_id")["sales_quantity"].sum().reset_index()
        fig = px.bar(by_store, x="store_id", y="sales_quantity", color="store_id",
                     labels={"store_id": "Store", "sales_quantity": "Units sold"},
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Promotion analysis")
        promo = filtered.copy()
        promo["Promotion"] = promo["promotion"].map({1: "Promotion", 0: "No promotion"})
        avg = promo.groupby("Promotion")["sales_quantity"].mean().reset_index()
        fig = px.bar(avg, x="Promotion", y="sales_quantity", color="Promotion",
                     labels={"sales_quantity": "Average units sold"},
                     color_discrete_sequence=["#81b29a", "#e07a5f"])
        fig.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if len(avg) == 2:
            lift = avg["sales_quantity"].max() / max(avg["sales_quantity"].min(), 1e-9)
            st.caption(f"Promotion days sell about **{lift:.2f}×** more units on average.")

    with st.expander("View the underlying data (first 500 rows)"):
        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
        download_button(filtered, "filtered_sales.csv", "⬇️ Download filtered sales as CSV")


# ----------------------------------------------------------------------
# PAGE 3: DEMAND FORECASTING
# ----------------------------------------------------------------------
def page_forecasting() -> None:
    page_header("🔮 Demand Forecasting", "Next 7 days of predicted demand from outputs/forecasts.csv")

    forecasts = load_csv(str(FORECAST_FILE))
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    left, mid, right = st.columns(3)
    stores = ["All stores"] + sorted(forecasts["store_id"].unique())
    selected_store = left.selectbox("Store", stores)

    products = sorted(
        forecasts.loc[forecasts["store_id"] == selected_store, "product_id"].unique()
        if selected_store != "All stores"
        else forecasts["product_id"].unique()
    )
    product_options = ["All products"] + list(products)
    selected_product = mid.selectbox("Product", product_options)
    metric_choice = right.radio("Chart view", ["Daily total", "By store-product"], horizontal=True)

    selected_forecasts = forecasts.copy()
    if selected_store != "All stores":
        selected_forecasts = selected_forecasts[selected_forecasts["store_id"] == selected_store]
    if selected_product != "All products":
        selected_forecasts = selected_forecasts[selected_forecasts["product_id"] == selected_product]

    if selected_forecasts.empty:
        st.warning("No forecast rows for this combination.")
        return

    kpi_row(
        [
            ("Forecast Rows", f"{len(selected_forecasts):,}", "Selected store-product-days"),
            ("Total Predicted Demand", f"{int(selected_forecasts['predicted_demand'].sum()):,}", "Units over the 7-day horizon"),
            ("Average per Day", f"{selected_forecasts.groupby('date')['predicted_demand'].sum().mean():,.1f}", "Average predicted units per day"),
            ("Peak Day", f"{selected_forecasts.groupby('date')['predicted_demand'].sum().max():,}", "Highest predicted day in the selection"),
        ]
    )

    st.subheader("Predicted demand")
    if metric_choice == "Daily total":
        chart_data = selected_forecasts.groupby("date")["predicted_demand"].sum().reset_index()
        fig = px.line(chart_data, x="date", y="predicted_demand", markers=True,
                      labels={"date": "Forecast date", "predicted_demand": "Predicted units"})
        fig.update_traces(line_color=ACCENT, line_width=3, marker_size=9)
    else:
        selected_forecasts["series"] = (selected_forecasts["store_id"] + " / "
                                        + selected_forecasts["product_id"])
        fig = px.line(selected_forecasts, x="date", y="predicted_demand", color="series",
                      markers=True,
                      labels={"date": "Forecast date", "predicted_demand": "Predicted units",
                              "series": "Store / Product"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast table")
    table = selected_forecasts.sort_values(["date", "store_id", "product_id"])
    st.dataframe(table, use_container_width=True, hide_index=True, height=320)

    col1, col2 = st.columns(2)
    with col1:
        download_button(table, "forecast_selection.csv", "⬇️ Download this selection as CSV")
    with col2:
        download_button(forecasts, "forecasts.csv", "⬇️ Download all forecasts (CSV)")

    st.info(
        "**How the forecast is produced:** the model predicts day 1 from real history, then "
        "feeds that prediction back in to predict day 2, and so on (recursive forecasting). "
        "Accuracy is best for day 1 and weakens for later days because the model starts using "
        "its own predictions as inputs. Future promotions are assumed to be zero."
    )


# ----------------------------------------------------------------------
# PAGE 4: INVENTORY & REPLENISHMENT
# ----------------------------------------------------------------------
def page_replenishment() -> None:
    page_header(
        "🚚 Inventory & Replenishment",
        "Reorder quantity = max(0, expected lead-time demand + safety stock − current stock)",
    )

    reorder = load_csv(str(REORDER_FILE))
    lead_time = int(reorder["lead_time_days"].iloc[0])

    kpi_row(
        [
            ("Store-Products", f"{len(reorder)}", "Rows analysed"),
            ("Need Restock", f"{int(reorder['needs_reorder'].sum())}", "Reorder quantity > 0"),
            ("Total Units to Order", f"{int(reorder['recommended_reorder_quantity'].sum()):,}", "Sum of recommendations"),
            ("Lead Time", f"{lead_time} days", "Assumed supplier lead time"),
            ("Total Current Stock", f"{int(reorder['current_stock'].sum()):,}", "Latest stock snapshot"),
        ]
    )

    # ---- Filters ----
    left, mid = st.columns([1, 1])
    only_restock = left.checkbox("Show only products that need restocking", value=False)
    category_filter = mid.multiselect(
        "Filter by category", sorted(reorder["product_category"].unique()), default=[]
    )

    view = reorder.copy()
    if only_restock:
        view = view[view["recommended_reorder_quantity"] > 0]
    if category_filter:
        view = view[view["product_category"].isin(category_filter)]

    st.subheader("Replenishment recommendations")
    display_cols = ["store_id", "product_id", "product_category", "current_stock",
                    "expected_demand", "safety_stock", "recommended_reorder_quantity"]
    styled = view[display_cols].style.apply(
        lambda row: [
            "background-color: #ffe5e5" if row["recommended_reorder_quantity"] > 0 else ""
            for _ in row
        ],
        axis=1,
    ).format({"expected_demand": "{:.2f}", "safety_stock": "{:.2f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=380)
    st.caption("Pink rows require restocking (reorder quantity above zero).")

    download_button(view[display_cols], "reorder_recommendations.csv",
                    "⬇️ Download recommendations as CSV")

    # ---- Charts ----
    left, right = st.columns(2)
    with left:
        st.subheader("Stock vs recommended reorder (top 15)")
        top = view.head(15).copy()
        top["label"] = top["store_id"] + " / " + top["product_id"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top["label"], y=top["current_stock"],
                             name="Current stock", marker_color="#8ecae6"))
        fig.add_trace(go.Bar(x=top["label"], y=top["recommended_reorder_quantity"],
                             name="Recommended reorder", marker_color=ACCENT))
        fig.update_layout(barmode="group", height=400, xaxis_title="Store / Product",
                          yaxis_title="Units", margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Recommendations by category")
        by_cat = (view.groupby("product_category")["recommended_reorder_quantity"]
                  .sum().reset_index())
        fig = px.bar(by_cat, x="product_category", y="recommended_reorder_quantity",
                     color="product_category", color_discrete_sequence=px.colors.sequential.Reds,
                     labels={"product_category": "Category",
                             "recommended_reorder_quantity": "Units to order"})
        fig.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Worked example ----
    if not view.empty:
        st.subheader("Worked example")
        example = view.iloc[0]
        st.markdown(
            f"""
            **Store {example['store_id']} / Product {example['product_id']}**
            ({example['product_category']})

            | Component | Value |
            |---|---|
            | Current stock | {example['current_stock']:.0f} units |
            | Expected demand during {lead_time}-day lead time | {example['expected_demand']:.2f} units |
            | Safety stock (1.65 × σ × √{lead_time}) | {example['safety_stock']:.2f} units |
            | **Recommended reorder** | **{example['recommended_reorder_quantity']:.0f} units** |

            `max(0, {example['expected_demand']:.2f} + {example['safety_stock']:.2f} − {example['current_stock']:.0f})`
            `= {example['recommended_reorder_quantity']:.0f} units`
            """
        )


# ----------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE
# ----------------------------------------------------------------------
def page_model_performance() -> None:
    page_header("🎯 Model Performance", "All metrics come from the pipeline's test-set evaluation")

    evaluation = load_csv(str(EVALUATION_FILE))
    category_errors = load_csv(str(CATEGORY_ERROR_FILE))
    predictions = load_csv(str(TEST_PREDICTIONS_FILE))
    predictions["date"] = pd.to_datetime(predictions["date"])
    metadata = load_model_metadata()

    best_model = metadata.get("model_name", evaluation.iloc[0]["model"])
    best_row = evaluation.loc[evaluation["model"] == best_model].iloc[0]

    kpi_row(
        [
            ("Selected Model", best_model, "Chosen by lowest test RMSE"),
            ("MAE", f"{best_row['MAE']:.2f}", "Average absolute error in units"),
            ("RMSE", f"{best_row['RMSE']:.2f}", "Penalises large errors"),
            ("R²", f"{best_row['R2']:.4f}", "Share of sales variation explained"),
        ]
    )

    st.subheader("Model comparison (test set)")
    st.dataframe(
        evaluation.style.highlight_min(subset=["MAE", "RMSE"], color="#c7f9cc")
        .highlight_max(subset=["R2"], color="#c7f9cc"),
        use_container_width=True, hide_index=True,
    )

    # ---- Metric comparison chart ----
    melted = evaluation.melt(id_vars="model", value_vars=["MAE", "RMSE", "R2"],
                             var_name="Metric", value_name="Value")
    fig = px.bar(melted, x="model", y="Value", color="Metric", barmode="group",
                 facet_col="Metric", facet_col_wrap=3,
                 color_discrete_sequence=["#2a6f97", "#e63946", "#81b29a"])
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Actual vs predicted ----
    st.subheader("Actual vs predicted sales (test period)")
    left, mid, right = st.columns(3)
    model_options = [c.replace("_prediction", "").replace("_", " ").title()
                     for c in predictions.columns if c.endswith("_prediction")]
    model_options = [m for m in model_options]
    chosen = left.selectbox("Model", model_options,
                            index=max(0, len(model_options) - 1))
    prediction_col = chosen.lower().replace(" ", "_") + "_prediction"

    store = mid.selectbox("Store", sorted(predictions["store_id"].unique()))
    products = sorted(predictions.loc[predictions["store_id"] == store, "product_id"].unique())
    product = right.selectbox("Product", products)

    subset = predictions[
        (predictions["store_id"] == store) & (predictions["product_id"] == product)
    ].sort_values("date")

    if subset.empty:
        st.warning("No test records for this store-product combination.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset["date"], y=subset["actual_sales"],
                                 name="Actual sales", line=dict(color="#1d3557", width=2)))
        fig.add_trace(go.Scatter(x=subset["date"], y=subset[prediction_col],
                                 name=f"Predicted ({chosen})",
                                 line=dict(color=ACCENT, width=2, dash="dash")))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Units sold",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        sse = ((subset["actual_sales"] - subset[prediction_col]) ** 2).sum()
        ss_tot = ((subset["actual_sales"] - subset["actual_sales"].mean()) ** 2).sum()
        local_r2 = 1 - sse / ss_tot if ss_tot > 0 else 0
        st.caption(
            f"This single series: MAE "
            f"{subset['actual_sales'].sub(subset[prediction_col]).abs().mean():.2f} units, "
            f"R² {local_r2:.3f}. A single series can score worse or better than the global "
            f"figures — the table above is the honest overall summary."
        )

    # ---- Category-wise errors ----
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Category-wise errors")
        st.dataframe(category_errors, use_container_width=True, hide_index=True)
        worst = category_errors.iloc[0]
        st.caption(
            f"Highest absolute error: **{worst['product_category']}** "
            f"(MAE {worst['MAE']:.2f} units). High-volume categories naturally show larger "
            f"absolute errors, so compare MAPE as well."
        )
        download_button(category_errors, "category_errors.csv", "⬇️ Download as CSV")

    with right:
        fig = px.bar(category_errors, x="product_category", y="MAE", color="product_category",
                     color_discrete_sequence=px.colors.sequential.Reds,
                     labels={"product_category": "Category", "MAE": "MAE (units)"})
        fig.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Saved evaluation charts from outputs/charts/"):
        for chart_name, caption in [
            ("07_model_comparison.png", "Model comparison"),
            ("06_actual_vs_predicted.png", "Actual vs predicted"),
            ("09_category_wise_errors.png", "Category-wise errors"),
        ]:
            chart_path = CHARTS_DIR / chart_name
            if chart_path.exists():
                st.image(str(chart_path), caption=caption, use_container_width=True)


# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
def main() -> None:
    # ---- Sidebar ----
    with st.sidebar:
        st.title("📦 Retail Forecasting")
        st.caption("Demand forecasting and replenishment dashboard")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Overview", "Sales Analysis", "Demand Forecasting",
             "Inventory & Replenishment", "Model Performance"],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Pipeline")
        st.caption("Outputs are read from disk. The model is not retrained on refresh.")
        if st.button("▶️ Run ML pipeline (main.py)", use_container_width=True):
            run_pipeline()
        if st.button("🔄 Reload outputs", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.caption(
            "**Dataset:** synthetic retail data "
            "(generated by Python, not real business data)."
        )
        model_exists = MODEL_FILE.exists()
        st.caption(f"**Model file:** {'found ✅' if model_exists else 'missing ❌'}")

    # ---- Missing outputs guard ----
    if not required_files_exist():
        st.title("📦 Retail Demand Forecasting & Replenishment")
        st.error(
            "Output files are missing. The dashboard reads the results produced by the "
            "pipeline, so run the pipeline first."
        )
        st.code("python main.py", language="bash")
        st.write("Missing files:")
        for path in [CLEANED_FILE, FORECAST_FILE, EVALUATION_FILE,
                     CATEGORY_ERROR_FILE, REORDER_FILE, TEST_PREDICTIONS_FILE]:
            st.write(f"- {'✅' if path.exists() else '❌'} `{path.relative_to(BASE_DIR)}`")
        if st.button("▶️ Run the pipeline now", type="primary"):
            run_pipeline()
        return

    # ---- Route to the selected page ----
    pages = {
        "Overview": page_overview,
        "Sales Analysis": page_sales_analysis,
        "Demand Forecasting": page_forecasting,
        "Inventory & Replenishment": page_replenishment,
        "Model Performance": page_model_performance,
    }
    try:
        pages[page]()
    except Exception as error:
        st.error(f"Something went wrong while drawing this page: {error}")
        st.info("Try 'Reload outputs' in the sidebar, or re-run the pipeline.")


if __name__ == "__main__":
    main()
