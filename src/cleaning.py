"""Cleaning logic for transaction-level data."""
from __future__ import annotations

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction data and add revenue/behavioral flags.

    Policy:
    - Drop rows with missing CustomerID.
    - Drop rows with UnitPrice <= 0.
    - Flag cancellations and return lines.
    - Compute line_revenue for valid purchase lines only.
    """
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["CustomerID"]).copy()
    cleaned = cleaned[cleaned["UnitPrice"] > 0].copy()

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned["is_cancellation"] = cleaned["InvoiceNo"].str.startswith("C")
    cleaned["is_return_line"] = cleaned["Quantity"] < 0

    valid_purchase = (~cleaned["is_cancellation"]) & (~cleaned["is_return_line"]) & (
        cleaned["Quantity"] > 0
    )
    cleaned["line_revenue"] = 0.0
    cleaned.loc[valid_purchase, "line_revenue"] = (
        cleaned.loc[valid_purchase, "Quantity"]
        * cleaned.loc[valid_purchase, "UnitPrice"]
    )
    return cleaned
