"""Data ingestion and audit utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src import config

STANDARD_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_transactions_excel(path: str | None = None) -> pd.DataFrame:
    """Load the Online Retail II Excel file, concatenating all sheets.

    Args:
        path: Optional override path to the Excel file.

    Returns:
        DataFrame with standardized columns and a `source_sheet` column.
    """
    excel_path = path or str(config.RAW_DATA_PATH)
    sheet_map: Dict[str, pd.DataFrame] = pd.read_excel(excel_path, sheet_name=None)
    frames = []
    for sheet_name, df in sheet_map.items():
        df = df.copy()
        df["source_sheet"] = sheet_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    combined = combined.rename(columns=str.strip)
    combined = combined.rename(columns={"Customer ID": "CustomerID"})

    # Map common alternate column names to the STANDARD_COLUMNS
    col_candidates = {c: c for c in combined.columns}
    alt_map = {
        "InvoiceNo": ["InvoiceNo", "Invoice No", "Invoice"],
        "StockCode": ["StockCode", "Stock Code"],
        "Description": ["Description", "ProductDescription"],
        "Quantity": ["Quantity", "Qty"],
        "InvoiceDate": ["InvoiceDate", "Invoice Date", "Date"],
        "UnitPrice": ["UnitPrice", "Unit Price", "Price"],
        "CustomerID": ["CustomerID", "Customer ID", "Customer Id"],
        "Country": ["Country", "country"],
    }

    rename_map = {}
    for target, options in alt_map.items():
        for opt in options:
            if opt in combined.columns:
                rename_map[opt] = target
                break

    if rename_map:
        combined = combined.rename(columns=rename_map)

    missing = [c for c in STANDARD_COLUMNS if c not in combined.columns]
    if missing:
        raise KeyError(f"Required columns not found in Excel sheets: {missing}. Available columns: {list(combined.columns)}")

    combined = combined[STANDARD_COLUMNS + ["source_sheet"]]
    combined["InvoiceDate"] = pd.to_datetime(combined["InvoiceDate"], errors="coerce")
    return combined


def audit_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Generate audit summary statistics for the dataset.

    Returns:
        A DataFrame summary and a dict of key metrics.
    """
    summary = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
            "n_unique": df.nunique(dropna=False),
        }
    )
    metrics = {
        "rows": len(df),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "duplicate_rows": df.duplicated().sum(),
        "total_customers": df["CustomerID"].nunique(dropna=True),
    }
    return summary, metrics


def save_raw_parquet(df: pd.DataFrame) -> str:
    """Save the consolidated raw dataset to parquet."""
    path = config.PROCESSED_DIR / "transactions_raw.parquet"
    # Ensure object-typed columns are converted to string to avoid pyarrow
    # attempting to coerce mixed-type columns to numeric types.
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("string")
    df.to_parquet(path, index=False)
    return str(path)
