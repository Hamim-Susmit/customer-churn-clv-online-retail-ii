"""Run final value-proxy and action list steps from notebooks/04.

This script loads cleaned transactions and customer features produced
by earlier notebooks, trains churn and value models, predicts scores,
and writes `reports/customer_action_list.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.features import (
    compute_future_revenue,
    build_time_based_labels,
)
from src.modeling import (
    select_feature_columns,
    build_churn_models,
    build_value_models,
    predict_churn_probabilities,
)


def main():
    processed = Path(config.PROCESSED_DIR)
    reports = Path(config.REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)

    clean_df = pd.read_parquet(processed / "transactions_clean.parquet")
    features = pd.read_parquet(processed / "customer_features.parquet")

    snapshot_date = pd.to_datetime(features["snapshot_date"]).max()
    cutoff_date = snapshot_date - pd.Timedelta(days=config.TIME_SPLIT.cutoff_days_before_snapshot)

    numeric_features, categorical_features = select_feature_columns(features)
    X = features[numeric_features + categorical_features]

    # Build churn labels using time-based split
    labels = build_time_based_labels(clean_df, cutoff_date, snapshot_date)
    # Align labels to features by CustomerID
    labels_map = labels.to_dict()
    y = features["CustomerID"].map(labels_map).astype(int)

    churn_model = build_churn_models(numeric_features, categorical_features)["hist_gradient_boosting"]
    churn_model.fit(X, y)
    features["churn_probability"] = predict_churn_probabilities(churn_model, X)

    # Future revenue and value model
    future_revenue = compute_future_revenue(clean_df, cutoff_date, window_days=config.DEFAULT_PREDICTION_WINDOW_DAYS)
    features = features.merge(future_revenue, on="CustomerID", how="left")
    rev_col = f"future_revenue_{config.DEFAULT_PREDICTION_WINDOW_DAYS}d"
    features[rev_col] = features[rev_col].fillna(0)

    value_model = build_value_models(numeric_features, categorical_features)["ridge"]
    value_model.fit(X, features[rev_col])
    features["value_score"] = value_model.predict(X)

    out_cols = ["CustomerID", "churn_probability", "value_score", rev_col]
    out_path = reports / "customer_action_list.csv"
    features[out_cols].to_csv(out_path, index=False)
    print(f"Wrote action list to {out_path}")


if __name__ == "__main__":
    main()
