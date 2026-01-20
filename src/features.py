"""Feature engineering for customer-level modeling."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src import config


def _invoice_totals(purchases: pd.DataFrame) -> pd.DataFrame:
    invoice_totals = (
        purchases.groupby(["CustomerID", "InvoiceNo"], as_index=False)["line_revenue"]
        .sum()
        .rename(columns={"line_revenue": "invoice_revenue"})
    )
    return invoice_totals


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level features from cleaned transactions."""
    snapshot_date = df["InvoiceDate"].max()
    purchases = df[(~df["is_cancellation"]) & (~df["is_return_line"])].copy()

    first_last = purchases.groupby("CustomerID")["InvoiceDate"].agg(
        first_purchase_date="min", last_purchase_date="max"
    )
    first_last["recency_days"] = (snapshot_date - first_last["last_purchase_date"]).dt.days
    first_last["tenure_days"] = (
        first_last["last_purchase_date"] - first_last["first_purchase_date"]
    ).dt.days

    num_invoices = purchases.groupby("CustomerID")["InvoiceNo"].nunique()
    first_last["num_invoices"] = num_invoices
    first_last["frequency_per_month"] = num_invoices / np.maximum(
        first_last["tenure_days"] / 30, 1
    )

    total_revenue = purchases.groupby("CustomerID")["line_revenue"].sum()
    first_last["total_revenue"] = total_revenue
    first_last["avg_order_value"] = total_revenue / np.maximum(num_invoices, 1)

    invoice_totals = _invoice_totals(purchases)
    median_order_value = invoice_totals.groupby("CustomerID")["invoice_revenue"].median()
    first_last["median_order_value"] = median_order_value
    first_last["revenue_per_month_active"] = total_revenue / np.maximum(
        first_last["tenure_days"] / 30, 1
    )

    first_last["unique_products"] = purchases.groupby("CustomerID")["StockCode"].nunique()
    first_last["unique_descriptions"] = purchases.groupby("CustomerID")["Description"].nunique()
    first_last["country_mode"] = (
        purchases.groupby("CustomerID")["Country"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)
    )

    total_lines = df.groupby("CustomerID").size()
    return_lines = df[df["is_return_line"]].groupby("CustomerID").size()
    cancellation_invoices = (
        df[df["is_cancellation"]].groupby("CustomerID")["InvoiceNo"].nunique()
    )
    total_invoices = df.groupby("CustomerID")["InvoiceNo"].nunique()

    first_last["return_line_rate"] = (return_lines / total_lines).fillna(0)
    first_last["cancellation_invoice_rate"] = (cancellation_invoices / total_invoices).fillna(0)
    first_last["net_quantity"] = purchases.groupby("CustomerID")["Quantity"].sum()

    avg_days_between = (
        purchases.sort_values(["CustomerID", "InvoiceDate"])
        .groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.days
        .groupby(purchases["CustomerID"])
        .mean()
    )
    first_last["avg_days_between_purchases"] = avg_days_between
    first_last["snapshot_date"] = snapshot_date

    return first_last.reset_index()


def add_churn_labels(features: pd.DataFrame, thresholds: Iterable[int] | None = None) -> pd.DataFrame:
    """Append churn labels for each threshold to the feature frame."""
    thresholds = thresholds or config.CHURN_THRESHOLDS_DAYS
    labeled = features.copy()
    for days in thresholds:
        labeled[f"churned_{days}d"] = (labeled["recency_days"] > days).astype(int)
    return labeled


def churn_sensitivity_table(features: pd.DataFrame, thresholds: Iterable[int] | None = None) -> pd.DataFrame:
    """Compute churn rate for each threshold."""
    thresholds = thresholds or config.CHURN_THRESHOLDS_DAYS
    rows = []
    for days in thresholds:
        churn_rate = (features["recency_days"] > days).mean()
        rows.append({"threshold_days": days, "churn_rate": churn_rate})
    return pd.DataFrame(rows)


def build_time_based_labels(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
) -> pd.Series:
    """Label customers based on purchases after cutoff date.

    churned = 1 if no purchases in (cutoff_date, snapshot_date]
    """
    purchases = df[(~df["is_cancellation"]) & (~df["is_return_line"])].copy()
    future_purchases = purchases[
        (purchases["InvoiceDate"] > cutoff_date) & (purchases["InvoiceDate"] <= snapshot_date)
    ]
    active_customers = set(future_purchases["CustomerID"].unique())
    customers = df["CustomerID"].drop_duplicates().values
    labels = [0 if cid in active_customers else 1 for cid in customers]
    return pd.Series(labels, index=customers, name="churned")


def compute_future_revenue(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: int = config.DEFAULT_PREDICTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Compute future revenue within a prediction window after cutoff date."""
    purchases = df[(~df["is_cancellation"]) & (~df["is_return_line"])].copy()
    window_end = cutoff_date + pd.Timedelta(days=window_days)
    window_data = purchases[
        (purchases["InvoiceDate"] > cutoff_date) & (purchases["InvoiceDate"] <= window_end)
    ]
    revenue = window_data.groupby("CustomerID")["line_revenue"].sum()
    return revenue.reset_index(name=f"future_revenue_{window_days}d")
